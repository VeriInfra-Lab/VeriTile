import bench.repairs.tritonbench_g.attention_score.Proofs.NativeInstructions
import bench.repairs.tritonbench_g.attention_score.Proofs.FPEngine

namespace VeriTile.Bench.Repairs.AttentionScore.NativeStaging
open VeriTile Triton FP.Structural NativeInstructions
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000

/-- Pending expressions represent compiler values before their instruction
consumer. Loads, reductions, exp2 and stores materialize a fixed native graph. -/
inductive Expr where
  | ready (graph : Graph)
  | literal (value : ℝ)
  | add (left right : Expr)
  | sub (left right : Expr)
  | mul (left right : Expr)
  | pendingDot (D row col : Nat) (q k : Fin D → Expr)

instance : Inhabited Expr := ⟨.ready zero⟩

/-- The observed fused instruction consumes scale first, dot second, and
the sign-bit-negated M word as its addend. No rounded mul/sub equality is used. -/
def lower : Expr → Graph
  | .ready g => g
  | .literal x => .literal32 x
  | .add seed (.pendingDot D row col q k) =>
      dot D row col (fun d => lower (q d)) (fun d => lower (k d)) (lower seed)
  | .add a b => .addRN (lower a) (lower b)
  | .sub (.mul a b) m => .fmaRN (lower b) (lower a) (.negateSign (lower m))
  | .sub a b => .addRN (lower a) (.negateSign (lower b))
  | .mul a b => .mulRN (lower a) (lower b)
  | .pendingDot D row col q k =>
      dot D row col (fun d => lower (q d)) (fun d => lower (k d)) zero

def planFor (width : Nat) : NativeReduction.Tree :=
  if width=16 then NativeReduction.plan ⟨0,by decide⟩ else
  if width=32 then NativeReduction.plan ⟨1,by decide⟩ else
  if width=64 then NativeReduction.plan ⟨2,by decide⟩ else .zero

theorem planFor_width (profile : Fin 3) :
    planFor (NativeReduction.width profile)=NativeReduction.plan profile := by
  fin_cases profile <;> rfl

def reductionInputs {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : Values Expr .real shape) (i : TileIndex (TileShape.reduceShape shape ax keep))
    (row : Fin (TileShape.axisDim shape ax)) : Expr :=
  match keep with
  | Bool.false => xs (TileShape.insertAxisIndex shape ax i row)
  | Bool.true => xs (TileShape.replaceAxisIndex shape ax i row)

def reduction (_ : Option ComputeDType) {shape : TileShape} (ax : Fin shape.length)
    (keep : Bool) (xs : Values Expr .real shape)
    (i : TileIndex (TileShape.reduceShape shape ax keep)) : Expr :=
  .ready ((planFor (TileShape.axisDim shape ax)).value zero
    (fun row => if h : row<TileShape.axisDim shape ax then lower (reductionInputs ax keep xs i ⟨row,h⟩) else zero)
    Graph.move Graph.addRN)

/-- This source-specific symbolic algebra defines exactly the floating
operations present in the repaired source: add/sub/mul, exp2, FP32 loads and
FP32 output casts. Other algebra fields are unreachable in this source; their
defaults do not represent additional native instruction claims. -/
def algebra : Algebra Expr where
  literal := fun _ _ x => .literal x
  negInf := .ready (.bits 0xFF800000)
  binary := fun _ _ op a b => match op with
    | .add => .add a b
    | .sub => .sub a b
    | .mul => .mul a b
    | _ => default
  unary := fun _ op a => match op with
    | .exp2 => .ready (.ex2ApproxFTZ (lower a))
    | _ => default
  cast := fun _ _ _ a => .ready (lower a)
  fromNat := fun _ n => .literal n
  fromInt := fun _ n => .literal n
  fp32Bits := fun b => .ready (.bits b.bits)
  fp32Load := fun a => .ready (lower a)
  reduceSum := reduction
  reduceMax := fun _ {_} _ _ _ _ => default

def dotValues (_ : Option ComputeDType) (left right out : FloatDType) (_ : DotOptions) :
    (batch : TileShape) → {m k n : Nat} →
      Values Expr left.toTileDType (batch++[m,k]) →
      Values Expr right.toTileDType (batch++[k,n]) →
      Values Expr out.toTileDType (batch++[m,n])
  | [], _, k, _, a, b, i => ofFloat out (.pendingDot k i.1.val i.2.1.val
      (fun d => toFloat left (a ⟨i.1,d,()⟩)) (fun d => toFloat right (b ⟨d,i.2.1,()⟩)))
  | _::ds, _, _, _, a, b, i => dotValues none left right out {} ds
      (fun j => a ⟨i.1,j⟩) (fun j => b ⟨i.1,j⟩) i.2

def dotModel : FPEngine.Dot Expr where
  apply := fun p l r out options {batch} {_} {_} {_} a b => dotValues p l r out options batch a b

theorem consume_dot (D row col : Nat) (q k : Fin D → Expr) (seed : Expr) :
    lower (.add seed (.pendingDot D row col q k))=
      dot D row col (fun d => lower (q d)) (fun d => lower (k d)) (lower seed) := by
  cases seed <;> rfl

theorem fused_scale_subtract (score scale normalizer : Expr) :
    lower (.sub (.mul score scale) normalizer)=
      .fmaRN (lower scale) (lower score) (.negateSign (lower normalizer)) := rfl

theorem reduction_coverage (profile : Fin 3) :
    (planFor (NativeReduction.width profile)).leaves.Perm (List.range (NativeReduction.width profile)) := by
  rw [planFor_width]
  exact NativeReduction.coverage profile

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScore.NativeStaging
