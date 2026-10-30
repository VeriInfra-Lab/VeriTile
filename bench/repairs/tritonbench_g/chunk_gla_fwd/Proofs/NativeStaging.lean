import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeInstructions

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging
open VeriTile Triton FP.Structural
open NativeInstructions
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
noncomputable section

/-- These are compiler values, not separately rounded numbers. Pending dot
and multiply values may be consumed by the immediately following addition.
Materialization at load/cast/shuffle boundaries prevents cross-boundary fusion. -/
inductive Expr where
  | bits (word : BitVec 32)
  | literal (precision : Option ComputeDType) (dtype : FloatDType) (value : ℝ)
  | binary (precision : Option ComputeDType) (dtype : FloatDType) (op : Binary) (a b : Expr)
  | unary (precision : Option ComputeDType) (op : Unary) (a : Expr)
  | cast (precision : Option ComputeDType) (source target : FloatDType) (a : Expr)
  | load (a : Expr)
  | move (route : NativeReduction.Route) (a : Expr)
  | pendingDot (precision : Option ComputeDType) (left right out : FloatDType)
      (options : DotOptions) (m k n row col : Nat) (a b : Fin k → Expr)

instance : Inhabited Expr := ⟨.bits 0⟩

def defaultTF32 (o : DotOptions) : Bool :=
  o.allowTF32 != some Bool.false && o.inputPrecision != some "ieee"

def lowerDot (o : DotOptions) (k row col : Nat) (a b : Fin k → Graph) (seed : Graph) :=
  if defaultTF32 o then tf32Dot k row col a b seed else ieeeDot k a b seed

/-- Structural lowering, with the actual default fusion retained. Every dot
exit expands into fixed instruction nodes. The exp implementation preserves
the observed FP32 log2(e) word followed by ex2.approx.f32. -/
def lower : Expr → Graph
  | .bits b => .bits b
  | .literal p d x => .literal p d x
  | .binary _ _ .add acc (.pendingDot _ _ _ _ o _ k _ row col a b) =>
      lowerDot o k row col (fun j => lower (a j)) (fun j => lower (b j)) (lower acc)
  | .binary _ _ .add (.binary _ _ .mul a b) c => .fma (lower a) (lower b) (lower c)
  | .binary _ _ .add a (.binary _ _ .mul b c) => .fma (lower b) (lower c) (lower a)
  | .binary p d op a b => .binary p d op (lower a) (lower b)
  | .unary p .exp a => .unary p .exp2 (.binary p .real .mul (lower a) (.bits 0x3FB8AA3B))
  | .unary p op a => .unary p op (lower a)
  | .cast p s t a => .cast p s t (lower a)
  | .load a => .move .register (lower a)
  | .move route a => .move route (lower a)
  | .pendingDot _ _ _ _ o _ k _ row col a b =>
      lowerDot o k row col (fun j => lower (a j)) (fun j => lower (b j)) zero

def planFor (n : Nat) (w : Fin 4) : NativeReduction.Tree :=
  match n with
  | 1 => NativeReduction.plan ⟨0,by decide⟩ w
  | 2 => NativeReduction.plan ⟨1,by decide⟩ w
  | 4 => NativeReduction.plan ⟨2,by decide⟩ w
  | 8 => NativeReduction.plan ⟨3,by decide⟩ w
  | 16 => NativeReduction.plan ⟨4,by decide⟩ w
  | 32 => NativeReduction.plan ⟨5,by decide⟩ w
  | 64 => NativeReduction.plan ⟨6,by decide⟩ w
  | 128 => NativeReduction.plan ⟨7,by decide⟩ w
  | 256 => NativeReduction.plan ⟨8,by decide⟩ w
  | n => NativeReduction.registerTree (List.range n)

theorem planFor_power (e : Fin 9) (w : Fin 4) :
    planFor (2^e.val) w=NativeReduction.plan e w := by fin_cases e <;> rfl

def reductionInputs {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : Values Expr .real shape) (i : TileIndex (TileShape.reduceShape shape ax keep))
    (k : Fin (TileShape.axisDim shape ax)) : Expr :=
  match keep with
  | Bool.false => xs (TileShape.insertAxisIndex shape ax i k)
  | Bool.true => xs (TileShape.replaceAxisIndex shape ax i k)

def reduction (w : Fin 4) (p : Option ComputeDType) {shape : TileShape}
    (ax : Fin shape.length) (keep : Bool) (xs : Values Expr .real shape)
    (i : TileIndex (TileShape.reduceShape shape ax keep)) : Expr :=
  (planFor (TileShape.axisDim shape ax) w).value (.literal p .real 0)
    (fun k => if h : k<TileShape.axisDim shape ax then reductionInputs ax keep xs i ⟨k,h⟩ else .literal p .real 0)
    Expr.move (Expr.binary p .real .add)

def algebra (w : Fin 4) : Algebra Expr where
  literal := Expr.literal
  negInf := .bits 0xFF800000
  binary := Expr.binary
  unary := Expr.unary
  cast := Expr.cast
  fromNat := fun p n => .literal p .real n
  fromInt := fun p n => .literal p .real n
  fp32Bits := fun b => .bits b.bits
  fp32Load := Expr.load
  reduceSum := reduction w
  reduceMax := fun p {shape} ax keep xs i =>
    (planFor (TileShape.axisDim shape ax) w).value (.bits 0xFF800000)
      (fun k => if h : k<TileShape.axisDim shape ax then reductionInputs ax keep xs i ⟨k,h⟩ else .bits 0xFF800000)
      Expr.move (Expr.binary p .real .max)

def dotValues (p : Option ComputeDType) (l r out : FloatDType) (o : DotOptions) :
    (batch : TileShape) → {m k n : Nat} →
      Values Expr l.toTileDType (batch++[m,k]) →
      Values Expr r.toTileDType (batch++[k,n]) →
      Values Expr out.toTileDType (batch++[m,n])
  | [], m, k, n, a, b, i => ofFloat out (.pendingDot p l r out o m k n i.1.val i.2.1.val
      (fun e => toFloat l (a ⟨i.1,e,()⟩)) (fun e => toFloat r (b ⟨e,i.2.1,()⟩)))
  | _::ds, _, _, _, a, b, i => dotValues p l r out o ds (fun j => a ⟨i.1,j⟩) (fun j => b ⟨i.1,j⟩) i.2

def dot : FPOutput.DotPrimitive Expr where
  apply := fun p l r out o {batch} {_} {_} {_} a b => dotValues p l r out o batch a b

theorem default_precision : defaultTF32 {}=Bool.true := rfl
theorem ieee_precision : defaultTF32 {allowTF32:=some Bool.false}=Bool.false := rfl

/-- A source carry plus pending dot becomes a seeded instruction chain by
definition. No separately rounded dot/add equality is used. -/
theorem consume_dot (p : Option ComputeDType) (l r out : FloatDType) (o : DotOptions)
    (m k n row col : Nat) (a b : Fin k → Expr) (seed : Expr) :
    lower (.binary p out .add seed (.pendingDot p l r out o m k n row col a b))=
      lowerDot o k row col (fun j => lower (a j)) (fun j => lower (b j)) (lower seed) := by
  cases seed <;> rfl

theorem reduction_leaf_coverage (e : Fin 9) (w : Fin 4) :
    (planFor (2^e.val) w).leaves.Perm (List.range (2^e.val)) := by
  rw [planFor_power]
  exact NativeReduction.coverage e w

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging
