import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelForwardGraph
import bench.tritonbench_g.parallel_retention_attention.Proofs.Arithmetic

/-! Fixed local staging for the retained FP32 scalar-dot lowering. A dot is
kept as syntax until its consumer is known. An immediate accumulator addition
becomes the same dot with its live carry; no split-add numerical identity is
asserted. Every dot is then expanded into a finite, coverage-checked scalar
multiply/add/FMA tree. No field supplies a kernel, loop, tensor, or result. -/
namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.Native
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (DotPrimitive)
noncomputable section

inductive Expr (α : Type) where
  | input (value : α)
  | literal (value : ℝ)
  | fromNat (value : Nat)
  | fromInt (value : Int)
  | unary (op : Unary) (x : Expr α)
  | binary (op : Binary) (x y : Expr α)
  | pendingDot (batch : TileShape) (m k n : Nat)
      (a : TileIndex (batch++[m,k]) → Expr α)
      (b : TileIndex (batch++[k,n]) → Expr α)
      (i : TileIndex (batch++[m,n]))
  | dotAcc (batch : TileShape) (m k n : Nat)
      (a : TileIndex (batch++[m,k]) → Expr α)
      (b : TileIndex (batch++[k,n]) → Expr α)
      (carry : Expr α) (i : TileIndex (batch++[m,n]))

instance : Inhabited (Expr α) := ⟨.literal 0⟩

def consume : Binary → Expr α → Expr α → Expr α
  | .add,carry,.pendingDot batch m k n a b i => .dotAcc batch m k n a b carry i
  | op,a,b => .binary op a b

/-- This task uses FP32 throughout. Same-format loads and casts transport the
word unchanged. Unused reduction fields are unreachable in these kernels. -/
def algebra : Algebra (Expr α) where
  literal := fun _ _ x => .literal x
  negInf := .literal 0
  binary := fun _ _ => consume
  unary := fun _ => Expr.unary
  cast := fun _ _ _ => id
  fromNat := fun _ => Expr.fromNat
  fromInt := fun _ => Expr.fromInt
  fp32Bits := fun _ => .literal 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => .literal 0
  reduceSum := fun _ {_} _ _ _ _ => .literal 0

def dots : DotPrimitive (Expr α) where
  apply := fun _ _ {batch} {m k n} a b i => .pendingDot batch m k n a b i

/-- Elementary numerical ports only. Host DK**(-0.5) conversion is represented
by its literal input, separately from arithmetic performed inside the kernel. -/
structure Scalar (α : Type) where
  literal : ℝ → α
  fromNat : Nat → α
  fromInt : Int → α
  binary : Binary → α → α → α
  unary : Unary → α → α
  fma : α → α → α → α

def Scalar.ops (a : Scalar α) : Ops α where
  zero := a.literal 0
  one := a.literal 1
  negFive := a.literal (-5)
  fromNat := a.fromNat
  fromInt := a.fromInt
  add := a.binary .add
  sub := a.binary .sub
  mul := a.binary .mul
  fma := a.fma
  exp2 := a.unary .exp2
  log2 := a.unary .log2
  cast := id
  scale := fun n => a.literal ((Real.sqrt (n : ℝ))⁻¹)

/-- Finite instruction-tree data, indexed by matrix dimensions and lane.
Coverage is a structural property; this record contains no numerical port. -/
structure DotTrees where
  tree : (m k n row col : Nat) → DotPlan
  complete : ∀ m k n row col, (tree m k n row col).Covers k

def DotTrees.linear : DotTrees := ⟨fun _ k _ _ _ => DotPlan.linear k,
  fun _ k _ _ _ => DotPlan.linear_complete k⟩

def matrixLane (batch : TileShape) {m n : Nat} : TileIndex (batch++[m,n]) → Nat×Nat :=
  match batch with
  | [] => fun i => (i.1.val,i.2.1.val)
  | _::rest => fun i => matrixLane rest i.2

/-- Replace the final matrix coordinates while preserving batch coordinates. -/
def dotInputs (batch : TileShape) {m k n : Nat}
    (a : TileIndex (batch++[m,k]) → α) (b : TileIndex (batch++[k,n]) → α) :
    TileIndex (batch++[m,n]) → Fin k → α×α :=
  match batch with
  | [] => fun i e => (a (i.1,e,PUnit.unit), b (e,i.2.1,PUnit.unit))
  | _::rest => fun i e => dotInputs rest (fun j=>a (i.1,j)) (fun j=>b (i.1,j)) i.2 e

def dotEval (a : Scalar α) (plan : DotTrees) (batch : TileShape) (m k n : Nat)
    (x : TileIndex (batch++[m,k]) → α) (y : TileIndex (batch++[k,n]) → α)
    (carry : α) (i : TileIndex (batch++[m,n])) : α :=
  let inputs := fun e => if h:e<k then dotInputs batch x y i ⟨e,h⟩
    else (a.literal 0,a.literal 0)
  let lane := matrixLane batch i
  (plan.tree m k n lane.1 lane.2).eval a.ops (fun e=>(inputs e).1) (fun e=>(inputs e).2) carry

def evaluate (a : Scalar α) (plan : DotTrees) : Expr α → α
  | .input x => x
  | .literal x => a.literal x
  | .fromNat x => a.fromNat x
  | .fromInt x => a.fromInt x
  | .unary op x => a.unary op (evaluate a plan x)
  | .binary op x y => a.binary op (evaluate a plan x) (evaluate a plan y)
  | .pendingDot batch m k n x y i => dotEval a plan batch m k n
      (fun j=>evaluate a plan (x j)) (fun j=>evaluate a plan (y j)) (a.literal 0) i
  | .dotAcc batch m k n x y carry i => dotEval a plan batch m k n
      (fun j=>evaluate a plan (x j)) (fun j=>evaluate a plan (y j)) (evaluate a plan carry) i

theorem evaluate_live_carry (a : Scalar α) (plan : DotTrees)
    (batch : TileShape) (m k n : Nat)
    (x : TileIndex (batch++[m,k]) → Expr α) (y : TileIndex (batch++[k,n]) → Expr α)
    (carry : Expr α) (i : TileIndex (batch++[m,n])) :
    evaluate a plan (consume .add carry (.pendingDot batch m k n x y i)) =
      dotEval a plan batch m k n (fun j=>evaluate a plan (x j))
        (fun j=>evaluate a plan (y j)) (evaluate a plan carry) i := rfl

#axiomsClean evaluate_live_carry
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.Native
