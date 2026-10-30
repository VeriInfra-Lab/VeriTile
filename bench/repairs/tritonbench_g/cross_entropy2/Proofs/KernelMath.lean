import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Math

namespace VeriTile.Bench.Repairs.CrossEntropy2.KernelMath
noncomputable section

def lse (p : Parameters) (x : Nat → ℝ) (block : Nat) : ℝ :=
  CrossEntropy1.Math.blockLSE p.cols p.block block (Math.scaled p.logitScale x)

def sum (p : Parameters) (x : Nat → ℝ) (block : Nat) : ℝ :=
  CrossEntropy1.Math.blockSum p.cols p.block block (Math.scaled p.logitScale x)

def loss (p : Parameters) (x : Nat → ℝ) (target : Int) (block : Nat) : ℝ :=
  let L := lse p x block
  let s := Math.smoothing p.smoothing p.hasSmoothing
  if target=p.ignored then 0 else
    (if p.split then 0 else L)-s*sum p x block/p.totalClasses-
    (if (block*p.block : Int)≤target-p.classStart ∧
        target-p.classStart<(min p.cols ((block+1)*p.block) : Nat)
      then (1-s)*Math.scaled p.logitScale x (target-p.classStart).toNat else 0)+
    (if p.split then 0 else p.squareScale*L*L)

def zLoss (p : Parameters) (x : Nat → ℝ) (target : Int) (block : Nat) : ℝ :=
  if target=p.ignored then 0 else if p.split then 0 else p.squareScale*lse p x block*lse p x block

def gradient (p : Parameters) (x : Nat → ℝ) (target : Int) (L upstream : ℝ) (col : Nat) : ℝ :=
  Math.gradientWithLSE p.totalClasses p.classStart target p.ignored
    (Math.smoothing p.smoothing p.hasSmoothing) p.logitScale p.squareScale upstream L x col

/-- A split program's scalar result is precisely its independent local
cross-entropy contribution; normalization is outside the kernel theorem. -/
theorem split_loss (p : Parameters) (x : Nat → ℝ) (target : Int) (block : Nat)
    (split : p.split=Bool.true) :
    loss p x target block=Math.partialLoss p.cols p.totalClasses p.block block p.classStart target p.ignored
      (Math.smoothing p.smoothing p.hasSmoothing) p.logitScale x := by
  by_cases ignored : target=p.ignored
  · simp [loss,Math.partialLoss,ignored]
  · simp [loss,Math.partialLoss,ignored,split,sum]
    ring

/-- The one-block unsplit formula is ordinary scaled cross entropy over
all supplied columns, with an optional local target and LSE-square term. -/
theorem unsplit_loss (p : Parameters) (x : Nat → ℝ) (target : Int)
    (covered : p.cols≤p.block) (split : p.split=Bool.false) :
    loss p x target 0=Math.loss p.cols p.totalClasses p.classStart target p.ignored
      (Math.smoothing p.smoothing p.hasSmoothing) p.logitScale p.squareScale x := by
  have owner : (p.classStart≤target ∧ target-p.classStart<(p.cols : Int)) ↔
      p.classStart≤target ∧ target<p.classStart+(p.cols : Int) := by omega
  by_cases ignored : target=p.ignored
  · simp [loss,Math.loss,ignored]
  · simp [loss,Math.loss,ignored,split,lse,sum,Math.LSE,Math.targetTerm,
      CrossEntropy1.Math.block_zero_lse p.cols p.block covered,
      CrossEntropy1.Math.block_zero_sum p.cols p.block covered,Nat.min_eq_left covered,owner]
    ring

#specNonCircular loss avoiding [forwardSyntax,backwardSyntax,Source.forward,Source.backward]
#specNonCircular gradient avoiding [forwardSyntax,backwardSyntax,Source.forward,Source.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.KernelMath
