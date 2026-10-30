import bench.tritonbench_g.layernorm_fwd_triton.Kernels
import bench.tritonbench_g.layernorm_fwd_triton.Proofs.Mathematical_FPContract
import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_WholeReduction

/-! Ordered default-FP32 native recurrence. FMA is a primitive of its own.
The selected compiler may commute operands and erase division by specialized
one; those choices are data bound to the actual compilation, never numerical
identities on an opaque carrier. Reduction leaf coverage is a list permutation,
not an assumed reduction-result equation. -/
namespace VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
namespace Native
noncomputable section

abbrev SumTree := VeriTile.Bench.TritonBenchG.LayerNormFwd.Whole.SumTree 128

structure Ops (α : Type) where
  (zero one : α)
  count : Nat → α
  (add sub mul div : α → α → α)
  fma : α → α → α → α
  sqrt : α → α

/-- The source casts FP32 loads to FP32 and its result to X's FP32 dtype.
They are identity bit moves in this default profile; no other dtype is asserted. -/
def cast32 (a : α) : α := a

def ordered (swap : Bool) (f : α → α → α) (a b : α) : α :=
  if swap then f b a else f a b

inductive Carry where
  | fused
  | separate (swap : Bool)

def carry (ops : Ops α) (recipe : Carry) (z acc : α) : α :=
  match recipe with
  | .fused => ops.fma z z acc
  | .separate swap => ordered swap ops.add acc (ops.mul z z)

structure Plan (N : Nat) where
  (meanTree varianceTree : SumTree)
  meanComplete : meanTree.leaves.Perm (List.finRange 128)
  varianceComplete : varianceTree.leaves.Perm (List.finRange 128)
  (meanSwap epsSwap normalSwap : Bool)
  varianceCarry : Carry
  eraseUnitDivision : Bool
  eraseUnitValid : eraseUnitDivision=Bool.true → N=1

def divideCount (ops : Ops α) (p : Plan N) (a : α) : α :=
  if p.eraseUnitDivision then a else ops.div a (ops.count N)

def maskedX (ops : Ops α) (x : Nat → α) (N col : Nat) : α :=
  if col<N then cast32 (x col) else ops.zero

def meanStep (ops : Ops α) (p : Plan N) (x : Nat → α)
    (off : Nat) (acc : Fin 128 → α) : Fin 128 → α := fun lane =>
  ordered p.meanSwap ops.add (acc lane) (maskedX ops x N (off+lane.val))

def meanAcc (ops : Ops α) (p : Plan N) (x : Nat → α) : Fin 128 → α :=
  foldRange (meanStep ops p x) 0 N 128 (fun _ => ops.zero)

def mean (ops : Ops α) (p : Plan N) (x : Nat → α) : α :=
  divideCount ops p (p.meanTree.eval ops.add (meanAcc ops p x))

/-- The subtraction is computed before the source's tail selection. Its
inactive result has no observable effect; the square/carry receives +0. -/
def centered (ops : Ops α) (x : Nat → α) (N : Nat) (mu : α) (col : Nat) : α :=
  let subtraction := ops.sub (maskedX ops x N col) mu
  if col<N then subtraction else ops.zero

def varianceStep (ops : Ops α) (p : Plan N) (x : Nat → α) (mu : α)
    (off : Nat) (acc : Fin 128 → α) : Fin 128 → α := fun lane =>
  let z := centered ops x N mu (off+lane.val)
  carry ops p.varianceCarry z (acc lane)

def varianceAcc (ops : Ops α) (p : Plan N) (x : Nat → α) (mu : α) : Fin 128 → α :=
  foldRange (varianceStep ops p x mu) 0 N 128 (fun _ => ops.zero)

def variance (ops : Ops α) (p : Plan N) (x : Nat → α) (mu : α) : α :=
  divideCount ops p (p.varianceTree.eval ops.add (varianceAcc ops p x mu))

def inverseStd (ops : Ops α) (p : Plan N) (eps v : α) : α :=
  ops.div ops.one (ops.sqrt (ordered p.epsSwap ops.add v eps))

def output (ops : Ops α) (p : Plan N) (mu inv x w : α) : α :=
  cast32 (ops.mul (ordered p.normalSwap ops.mul (ops.sub (cast32 x) mu) inv) (cast32 w))

/-- Input-only FP target retaining the selected FMA, reduction trees, runtime
epsilon word, count conversion, signed zeros and unspecified primitive values. -/
def reference (ops : Ops α) (p : Plan N) (x w : Nat → α) (eps : α) (col : Nat) : α :=
  let mu := mean ops p x
  let v := variance ops p x mu
  output ops p mu (inverseStd ops p eps v) (x col) (w col)

end
end Native
end VeriTile.Bench.Repairs.LayernormFwdTriton
