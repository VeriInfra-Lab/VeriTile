import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelSource

/-! Independent ordered scalar and tile expressions for the selected source.
These definitions contain no source execution or final-state query. -/
namespace VeriTile.Bench.Repairs.FastCeLoss.KernelModel
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
variable {α : Type} [Inhabited α]

def literal (M : Algebra α) (x : ℝ) : α := M.literal (some .fp32) .real x
def cast (M : Algebra α) (x : α) : α := M.cast (some .fp32) .real .real x
def binary (M : Algebra α) (op : Binary) (x y : α) : α :=
  M.binary (some .fp32) .real op x y
def unary (M : Algebra α) (op : Unary) (x : α) : α := M.unary (some .fp32) op x

/-- Forward and backward preserve their original multiplication operand order. -/
def forwardScale (M : Algebra α) (p : Parameters) (x : α) : α :=
  if p.hasScaling then binary M .mul (literal M p.scaling) x else x

def backwardScale (M : Algebra α) (p : Parameters) (x : α) : α :=
  if p.hasScaling then binary M .mul x (literal M p.scaling) else x

def capPartial (M : Algebra α) (p : Parameters) (x : α) : α :=
  unary M .libdeviceTanh (binary M .div x (literal M p.softcap))

def cap (M : Algebra α) (p : Parameters) (x : α) : α :=
  if p.hasSoftcap then binary M .mul (literal M p.softcap) (capPartial M p x) else x

def loaded (M : Algebra α) (p : Parameters) (initial : State α)
    (col : Nat) : α :=
  M.fp32Load (if col<p.vocab then
    (initial.mem p.logits (initial.pids 0*p.logitsStride+col)).read .real else M.negInf)

def forwardLogits (M : Algebra α) (p : Parameters) (initial : State α)
    (chunk : Nat) : Values α .real [p.block] :=
  fun (lane,_) => if chunk*p.block+lane.val<p.vocab then
    cast M (cap M p (forwardScale M p (loaded M p initial (chunk*p.block+lane.val))))
    else M.negInf

def maxTile (M : Algebra α) (p : Parameters) (initial : State α)
    (chunk : Nat) : Values α .real [] :=
  M.reduceMax (some .fp32) ⟨0,by simp⟩ Bool.false (fun i => forwardLogits M p initial chunk i)

def maxValue (M : Algebra α) (p : Parameters) (initial : State α) (chunk : Nat) : α :=
  maxTile M p initial chunk PUnit.unit

def lseTile (M : Algebra α) (p : Parameters) (initial : State α)
    (chunk : Nat) : Values α .real [] :=
  fun _ => binary M .add (maxValue M p initial chunk)
    (unary M .log (M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false
      (fun i => unary M .exp (binary M .sub (forwardLogits M p initial chunk i)
        (maxValue M p initial chunk))) PUnit.unit))

def lseValue (M : Algebra α) (p : Parameters) (initial : State α) (chunk : Nat) : α :=
  lseTile M p initial chunk PUnit.unit

def gathered (M : Algebra α) (p : Parameters) (initial : State α) : α :=
  M.fp32Load ((initial.mem p.logits
    (initial.pids 0*p.logitsStride+(label p initial).toNat)).read .real)

def forwardLoss (M : Algebra α) (p : Parameters) (initial : State α) : α :=
  if label p initial = -100 then literal M 0 else
    binary M .sub (lseValue M p initial 0)
      (cast M (cap M p (forwardScale M p (gathered M p initial))))

def chunkTarget (M : Algebra α) (p : Parameters) (initial : State α) : α :=
  cast M ((initial.mem p.logits
    (initial.pids 0*p.logitsStride+(label p initial).toNat)).read .real)

def chunkLoss (M : Algebra α) (p : Parameters) (initial : State α) : α :=
  if label p initial = -100 then literal M 0 else
    binary M .mul (literal M (-1))
      (cast M (cap M p (forwardScale M p (chunkTarget M p initial))))

def upstreamValue (M : Algebra α) (p : Parameters) (initial : State α) : α :=
  if label p initial = -100 then literal M 0 else
    M.fp32Load ((initial.mem p.upstream (initial.pids 0*p.upstreamStride)).read .real)

def savedLSE (M : Algebra α) (p : Parameters) (initial : State α) : α :=
  M.fp32Load ((initial.mem p.lse (initial.pids 0)).read .real)

def backwardProbability (M : Algebra α) (p : Parameters) (initial : State α)
    (lane : Fin p.block) : α :=
  let col := initial.pids 1*p.block+lane.val
  let x := backwardScale M p (loaded M p initial col)
  unary M .exp (binary M .sub (cast M (cap M p x)) (savedLSE M p initial))

def backwardCorrected (M : Algebra α) (p : Parameters) (initial : State α)
    (lane : Fin p.block) : α :=
  let col := initial.pids 1*p.block+lane.val
  let y := backwardProbability M p initial lane
  if (col : Int)=label p initial then binary M .sub y (literal M 1) else y

def backwardDerivative (M : Algebra α) (p : Parameters) (initial : State α)
    (lane : Fin p.block) : α :=
  let col := initial.pids 1*p.block+lane.val
  let y := backwardScale M p (backwardCorrected M p initial lane)
  if p.hasSoftcap then
    let tanhValue := capPartial M p (backwardScale M p (loaded M p initial col))
    binary M .mul y (binary M .sub (literal M 1) (binary M .mul tanhValue tanhValue))
  else y

def gradientValue (M : Algebra α) (p : Parameters) (initial : State α)
    (lane : Fin p.block) : α :=
  cast M (if label p initial = -100 then literal M 0 else
    binary M .mul (upstreamValue M p initial) (backwardDerivative M p initial lane))

end
end VeriTile.Bench.Repairs.FastCeLoss.KernelModel
