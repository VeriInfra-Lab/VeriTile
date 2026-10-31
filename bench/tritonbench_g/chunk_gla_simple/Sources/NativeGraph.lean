import bench.tritonbench_g.chunk_gla_simple.Sources.FPGrid

namespace VeriTile.Bench.Repairs.ChunkGlaSimple.NativeGraph
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple Host
noncomputable section

/-- Native staging payloads, not already-rounded dot results. The input words
remain leaves and every gate/cast operation remains an explicit node. -/
inductive Expr where
  | leaf (word : BitVec 32)
  | literal (precision : Option ComputeDType) (dtype : FloatDType) (value : ℝ)
  | load (value : Expr)
  | cast (precision : Option ComputeDType) (source target : FloatDType) (value : Expr)
  | unary (precision : Option ComputeDType) (op : Unary) (value : Expr)
  | binary (precision : Option ComputeDType) (dtype : FloatDType) (op : Binary) (left right : Expr)
  | pendingDot (precision : Option ComputeDType) (options : DotOptions)
      (batch : TileShape) (m k n : Nat)
      (a : TileIndex (batch++[m,k]) → Expr) (b : TileIndex (batch++[k,n]) → Expr)
      (i : TileIndex (batch++[m,n]))
  | dotAcc (precision : Option ComputeDType) (options : DotOptions)
      (batch : TileShape) (m k n : Nat)
      (a : TileIndex (batch++[m,k]) → Expr) (b : TileIndex (batch++[k,n]) → Expr)
      (carry : Expr) (i : TileIndex (batch++[m,n]))
instance : Inhabited Expr := ⟨.leaf 0⟩

def options : DotOptions := {allowTF32 := some Bool.false}

/-- This is graph construction before numerical interpretation. It is not an
identity between a separately rounded dot/add and a fused native operation. -/
def consume (precision : Option ComputeDType) (dtype : FloatDType) : Binary → Expr → Expr → Expr
  | .add, carry, .pendingDot p opts batch m k n a b i => .dotAcc p opts batch m k n a b carry i
  | op, a, b => .binary precision dtype op a b

def algebra (rest : Algebra Expr) : Algebra Expr :=
  {rest with
    literal := Expr.literal
    fp32Load := Expr.load
    cast := Expr.cast
    unary := Expr.unary
    binary := consume
    dotWithOptions := fun p opts {batch} {m k n} a b => some (fun i => .pendingDot p opts batch m k n a b i)}

def dots (rest : Algebra Expr) (c : Shape) : Typed.Dots (algebra rest) c where
  qh := fun a b i => .pendingDot (some .fp32) options [] c.BT c.BK c.BV a b i
  qk := fun a b i => .pendingDot (some .fp32) options [] c.BT c.BK c.BT a b i
  sv := fun a b i => .pendingDot (some .fp32) options [] c.BT c.BT c.BV a b i
  qh_eq := fun _ _ => rfl
  qk_eq := fun _ _ => rfl
  sv_eq := fun _ _ => rfl

/-- Independent initial-memory recurrence forms the actual QH and QK dotAcc
nodes with their live carries at every K block. -/
theorem update_dotAcc (rest : Algebra Expr) (c : Shape) (s : State Expr)
    (block : Nat) (acc : FPValue.Acc Expr c) :
    FPValue.update (algebra rest) c (dots rest c) s block acc =
      (fun i => Expr.dotAcc (some .fp32) options [] c.BT c.BK c.BV
        (FPValue.qValue (algebra rest) c s block) (FPValue.hValue (algebra rest) c s block) (acc.1 i) i,
       fun i => Expr.dotAcc (some .fp32) options [] c.BT c.BK c.BT
        (FPValue.qValue (algebra rest) c s block) (FPValue.kValue (algebra rest) c s block) (acc.2 i) i) := rfl

theorem output_dotAcc (rest : Algebra Expr) (c : Shape) (s : State Expr) (scale : ℝ)
    (i : FPValue.OIndex c) :
    FPValue.outputValue (algebra rest) c (dots rest c) s scale i =
      Expr.cast (some .fp32) .real .fp32
        (.binary (some .fp32) .real .mul
          (.dotAcc (some .fp32) options [] c.BT c.BT c.BV
            (fun j => .cast (some .fp32) .real .real
              (FPValue.gatedS (algebra rest) c s (FPValue.accumulated (algebra rest) c (dots rest c) s) j))
            (FPValue.vValue (algebra rest) c s)
            (FPValue.gatedO (algebra rest) c s (FPValue.accumulated (algebra rest) c (dots rest c) s) i) i)
          (.literal (some .fp32) .real scale)) := rfl

/-- Exact local numerical ports for the staged FP32/IEEE dot graph. No
whole-output, execution, address, or split-rounded-equals-dotAcc law appears. -/
structure Primitives (α : Type) where
  leaf : BitVec 32 → α
  literal : Option ComputeDType → FloatDType → ℝ → α
  load : α → α
  cast : Option ComputeDType → FloatDType → FloatDType → α → α
  unary : Option ComputeDType → Unary → α → α
  binary : Option ComputeDType → FloatDType → Binary → α → α → α
  dot : Option ComputeDType → DotOptions → (batch : TileShape) → (m k n : Nat) →
    (TileIndex (batch++[m,k]) → α) → (TileIndex (batch++[k,n]) → α) → TileIndex (batch++[m,n]) → α
  dotAcc : Option ComputeDType → DotOptions → (batch : TileShape) → (m k n : Nat) →
    (TileIndex (batch++[m,k]) → α) → (TileIndex (batch++[k,n]) → α) → α → TileIndex (batch++[m,n]) → α

def evaluate {α : Type} (p : Primitives α) : Expr → α
  | .leaf w => p.leaf w
  | .literal q d v => p.literal q d v
  | .load x => p.load (evaluate p x)
  | .cast q a b x => p.cast q a b (evaluate p x)
  | .unary q op x => p.unary q op (evaluate p x)
  | .binary q d op x y => p.binary q d op (evaluate p x) (evaluate p y)
  | .pendingDot q opts batch m k n a b i => p.dot q opts batch m k n
      (fun j => evaluate p (a j)) (fun j => evaluate p (b j)) i
  | .dotAcc q opts batch m k n a b acc i => p.dotAcc q opts batch m k n
      (fun j => evaluate p (a j)) (fun j => evaluate p (b j)) (evaluate p acc) i

#axiomsClean update_dotAcc
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaSimple.NativeGraph
