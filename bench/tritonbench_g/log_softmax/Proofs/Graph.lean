import bench.tritonbench_g.layer_norm_fwd.Sources.Kernels

/-! The actual selected FP32 scalar/reduction graph. Leaves are all padded
lanes, and roots may differ by consumer lane. Native contraction is explicit;
no equality between a separate multiply/subtract and FMA is presumed. Source
and compiler profile pins bind this graph to the unchanged candidate functions.
Memory execution and the independent Real formula are separate helpers. -/
namespace VeriTile.Bench.Repairs.LogSoftmax
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.TritonBenchG.LayerNormFwd.Whole
noncomputable section

structure Ops (α : Type) where
  zero : α
  negativeInfinity : α
  (add subtract multiply maximum : α → α → α)
  (exp log negate : α → α)
  fma : α → α → α → α

inductive BackwardSchedule where
  | separate
  | fused
  | fusedScale
  deriving DecidableEq, Repr

structure Plan (N C : Nat) where
  maxTree : Nat → SumTree C
  forwardSum : Nat → SumTree C
  backwardSum : Nat → SumTree C
  maxComplete : ∀ lane, (maxTree lane).leaves.Perm (List.finRange C)
  forwardComplete : ∀ lane, (forwardSum lane).leaves.Perm (List.finRange C)
  backwardComplete : ∀ lane, (backwardSum lane).leaves.Perm (List.finRange C)
  covers : N≤C
  backwardSchedule : BackwardSchedule

/-- A complete ordered tree exists for every positive specification width.
This construction is used only to state an empty-return observation: it does
not assert compiler acceptance and is never executed by a native trial. -/
def specificationTree (first : Fin C) : List (Fin C) → SumTree C
  | []=>.leaf first
  | next::rest=>.add (.leaf first) (specificationTree next rest)

/-- The source's selected FP32 load and store casts are representation identities. -/
def cast32 (x : α) : α := x

def forwardLoad (ops : Ops α) (N : Nat) (x : Nat → α) (j : Fin C) : α :=
  if j.val<N then cast32 (x j.val) else ops.negativeInfinity

def rowMax (ops : Ops α) (plan : Plan N C) (x : Nat → α) (lane : Nat) : α :=
  (plan.maxTree lane).eval ops.maximum (forwardLoad ops N x)

def shifted (ops : Ops α) (plan : Plan N C) (x : Nat → α) (j : Fin C) : α :=
  ops.subtract (forwardLoad ops N x j) (rowMax ops plan x j.val)

def exponentials (ops : Ops α) (plan : Plan N C) (x : Nat → α) (j : Fin C) : α :=
  ops.exp (shifted ops plan x j)

def denominator (ops : Ops α) (plan : Plan N C) (x : Nat → α) (lane : Nat) : α :=
  (plan.forwardSum lane).eval ops.add (exponentials ops plan x)

/-- Stable source graph; it never computes log(exp(delta)/sum). -/
def forward (ops : Ops α) (plan : Plan N C) (x : Nat → α) (column : Nat) : α :=
  cast32 (ops.subtract (ops.subtract (cast32 (x column)) (rowMax ops plan x column))
    (ops.log (denominator ops plan x column)))

def gradientLoad (ops : Ops α) (N : Nat) (g : Nat → α) (j : Fin C) : α :=
  if j.val<N then cast32 (g j.val) else ops.zero

def scale (ops : Ops α) (plan : Plan N C) (g : Nat → α) (lane : Nat) : α :=
  (plan.backwardSum lane).eval ops.add (gradientLoad ops N g)

def backward (ops : Ops α) (plan : Plan N C) (y g : Nat → α) (column : Nat) : α :=
  let probability := ops.exp (cast32 (cast32 (y column)))
  let total := scale ops plan g column
  cast32 (match plan.backwardSchedule with
    | .separate => ops.subtract (cast32 (g column)) (ops.multiply probability total)
    | .fused => ops.fma (ops.negate probability) total (cast32 (g column))
    | .fusedScale => ops.fma (ops.negate total) probability (cast32 (g column)))

inductive Direction where
  | forward | backward
  deriving DecidableEq, Repr

def value (direction : Direction) (ops : Ops α) (plan : Plan N C)
    (first second : Nat → α) (column : Nat) : α :=
  match direction with
  | .forward => forward ops plan first column
  | .backward => backward ops plan first second column

end
end VeriTile.Bench.Repairs.LogSoftmax
