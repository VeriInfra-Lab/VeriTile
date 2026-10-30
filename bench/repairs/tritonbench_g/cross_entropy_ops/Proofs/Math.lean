import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Math
import bench.repairs.tritonbench_g.cross_entropy_ops.Kernels

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Spec
open scoped BigOperators
noncomputable section

/-- Independent logical row, including zero and negative logit scale. -/
def scaled (a : ℝ) (x : Nat → ℝ) : Nat → ℝ := fun j => x j*a
def lse (classes : Nat) (a : ℝ) (x : Nat → ℝ) : ℝ :=
  Real.log (∑ j ∈ Finset.range classes, Real.exp (x j*a))
def loss (classes : Nat) (a smoothing z : ℝ) (x : Nat → ℝ) (target ignored : Int) : ℝ :=
  if target=ignored then 0 else
    lse classes a x-(1-smoothing)*(x target.toNat*a)-
      smoothing*(∑ j ∈ Finset.range classes, x j*a)/classes+z*lse classes a x*lse classes a x
def zLoss (classes : Nat) (a z : ℝ) (x : Nat → ℝ) (target ignored : Int) : ℝ :=
  if target=ignored then 0 else z*lse classes a x*lse classes a x
def gradient (classes : Nat) (a smoothing z upstream : ℝ) (x : Nat → ℝ)
    (target ignored : Int) (col : Nat) : ℝ :=
  if target=ignored then 0 else
    (upstream*a)*((1+2*z*lse classes a x)*Real.exp (x col*a-lse classes a x)-
      (if (col : Int)=target then 1-smoothing else 0)-smoothing/classes)

theorem loss_reuses_ce1 (classes : Nat) (a smoothing z : ℝ) (x : Nat → ℝ) (target ignored : Int) :
    loss classes a smoothing z x target ignored=
      CrossEntropy1.Math.loss classes (scaled a x) target ignored smoothing z := rfl

theorem gradient_reuses_ce1 (classes : Nat) (a smoothing z upstream : ℝ) (x : Nat → ℝ)
    (target ignored : Int) (col : Nat) :
    gradient classes a smoothing z upstream x target ignored col=
      CrossEntropy1.Math.gradient classes (scaled a x) target ignored smoothing z (upstream*a) col := rfl

theorem local_lse_composition (classes width : Nat) (positive : 0<width) (a : ℝ) (x : Nat → ℝ) :
    Real.log (∑ block ∈ Finset.range (CrossEntropy1.Math.blocks classes width),
      Real.exp (CrossEntropy1.Math.blockLSE classes width block (scaled a x)))=lse classes a x := by
  exact CrossEntropy1.Math.local_lse_composition classes width positive (scaled a x)

theorem ignored_loss (classes : Nat) (a smoothing z : ℝ) (x : Nat → ℝ) (ignored : Int) :
    loss classes a smoothing z x ignored ignored=0 ∧ zLoss classes a z x ignored ignored=0 := by
  simp [loss,zLoss]

theorem ignored_gradient (classes : Nat) (a smoothing z upstream : ℝ) (x : Nat → ℝ)
    (ignored : Int) (col : Nat) : gradient classes a smoothing z upstream x ignored ignored col=0 := by
  simp [gradient]

theorem label_stride_counterexample :
    (fun i : Nat => if i=1 then (1 : Int) else 0) (1*2)=0 ∧
    (fun i : Nat => if i=1 then (1 : Int) else 0) 1=1 := by norm_num

#axiomsClean loss_reuses_ce1
#axiomsClean gradient_reuses_ce1
#axiomsClean local_lse_composition
#axiomsClean ignored_loss
#axiomsClean ignored_gradient
#axiomsClean label_stride_counterexample
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Spec
