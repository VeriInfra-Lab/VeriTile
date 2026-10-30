import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelRealBackward
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelReal
open VeriTile.Triton FP.Structural KernelRegisters KernelBackwardRegisters
open scoped BigOperators
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 14000000

def gradient (s : Shape) (scale : ℝ) (a : Fin s.BBV → Math.Input s.BK)
    (g : Nat → Fin s.BBV → ℝ) (terminal : Fin s.BBV → Math.Vec s.BK) (t : Nat) (j : Fin s.BBV) :=
  Math.gradient scale (a j) (fun n => g n j) (terminal j) s.T t
def enhanced (s : Shape) (scale : ℝ) (b : KernelBackwardReference.Input s ℝ) (t : Nat) :=
  KernelBackwardReference.enhance algebra s scale b t (KernelBackwardReference.reverse algebra s scale b (s.T-1-t))

theorem raw_value_dot (s : Shape) (scale : ℝ) (b : KernelBackwardReference.Input s ℝ)
    (a : Fin s.BBV → Math.Input s.BK) (g : Nat → Fin s.BBV → ℝ) (terminal : Fin s.BBV → Math.Vec s.BK)
    (saved : Saved s b a g terminal) (t : Nat) (j : Fin s.BBV) :
    KernelBackwardReference.rawDv algebra s b t (enhanced s scale b t) j=
      Math.dot (enhancedColumn s scale (a j) (fun n => g n j) (terminal j) t) ((a j).k t) := by
  unfold KernelBackwardReference.rawDv
  rw [key_sum]
  simp only [KernelReference.mul]
  change (∑ i : Fin s.BK,enhanced s scale b t (i,j,PUnit.unit)*b.key t i)=_
  simp_rw [enhanced,enhanced_column s scale b a g terminal saved,saved.key _ _ j]
  rfl

theorem value_gradient (s : Shape) (scale : ℝ) (b : KernelBackwardReference.Input s ℝ)
    (a : Fin s.BBV → Math.Input s.BK) (g : Nat → Fin s.BBV → ℝ) (terminal : Fin s.BBV → Math.Vec s.BK)
    (saved : Saved s b a g terminal) (t : Nat) (j : Fin s.BBV) :
    KernelBackwardReference.dv algebra s b t (enhanced s scale b t) j=(gradient s scale a g terminal t j).x := by
  change KernelBackwardReference.rawDv algebra s b t (enhanced s scale b t) j*b.beta t j=_
  rw [raw_value_dot s scale b a g terminal saved,saved.beta]
  simp only [gradient,Math.gradient,Math.pullback,Math.carry,enhancedColumn]
  have time : s.T-(t+1)=s.T-1-t := by omega
  rw [time]
  exact mul_comm _ _
theorem beta_gradient (s : Shape) (scale : ℝ) (b : KernelBackwardReference.Input s ℝ)
    (a : Fin s.BBV → Math.Input s.BK) (g : Nat → Fin s.BBV → ℝ) (terminal : Fin s.BBV → Math.Vec s.BK)
    (saved : Saved s b a g terminal) (t : Nat) (ht : t<s.T) (j : Fin s.BBV) :
    KernelBackwardReference.dbeta algebra s b t (enhanced s scale b t) j=(gradient s scale a g terminal t j).beta := by
  change KernelBackwardReference.rawDv algebra s b t (enhanced s scale b t) j*b.value t j=_
  rw [raw_value_dot s scale b a g terminal saved,saved.residual t ht]
  simp only [gradient,Math.gradient,Math.pullback,Math.carry,enhancedColumn]
  have time : s.T-(t+1)=s.T-1-t := by omega
  rw [time]
  exact mul_comm _ _
theorem scalar_beta_gradient (s : Shape) (scale : ℝ) (b : KernelBackwardReference.Input s ℝ)
    (a : Fin s.BBV → Math.Input s.BK) (g : Nat → Fin s.BBV → ℝ) (terminal : Fin s.BBV → Math.Vec s.BK)
    (saved : Saved s b a g terminal) (t : Nat) (ht : t<s.T) :
    KernelBackwardReference.betaSum algebra s (KernelBackwardReference.dbeta algebra s b t (enhanced s scale b t))=
      ∑ j : Fin s.BBV,(gradient s scale a g terminal t j).beta := by
  rw [beta_sum]
  simp_rw [beta_gradient s scale b a g terminal saved t ht]

theorem query_gradient (s : Shape) (scale : ℝ) (b : KernelBackwardReference.Input s ℝ)
    (a : Fin s.BBV → Math.Input s.BK) (g : Nat → Fin s.BBV → ℝ) (terminal : Fin s.BBV → Math.Vec s.BK)
    (saved : Saved s b a g terminal) (t : Nat) (ht : t<s.T) (i : Fin s.BK) :
    KernelBackwardReference.dq algebra s scale b t (KernelBackwardReference.forward algebra s b (t+1)) i=
      ∑ j : Fin s.BBV,(gradient s scale a g terminal t j).q i := by
  unfold KernelBackwardReference.dq
  rw [value_sum]
  change (∑ j : Fin s.BBV,KernelBackwardReference.forward algebra s b (t+1) (i,j,PUnit.unit)*b.incoming t j)*scale=_
  rw [Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro j _
  rw [recompute_state s b a g terminal saved (t+1) (by omega),saved.incoming]
  simp only [gradient,Math.gradient,Math.pullback,Math.state]
  ring

theorem key_gradient (s : Shape) (scale : ℝ) (b : KernelBackwardReference.Input s ℝ)
    (a : Fin s.BBV → Math.Input s.BK) (g : Nat → Fin s.BBV → ℝ) (terminal : Fin s.BBV → Math.Vec s.BK)
    (saved : Saved s b a g terminal) (t : Nat) (ht : t<s.T) (i : Fin s.BK) :
    KernelBackwardReference.correction algebra s (KernelBackwardReference.forward algebra s b t)
      (KernelBackwardReference.dk algebra s b t (enhanced s scale b t))
      (KernelBackwardReference.dv algebra s b t (enhanced s scale b t)) i=
      ∑ j : Fin s.BBV,(gradient s scale a g terminal t j).k i := by
  unfold KernelBackwardReference.correction KernelBackwardReference.dk
  rw [value_sum,value_sum]
  change (∑ j : Fin s.BBV,enhanced s scale b t (i,j,PUnit.unit)*(b.value t j*b.beta t j))-
    (∑ j : Fin s.BBV,KernelBackwardReference.dv algebra s b t (enhanced s scale b t) j*
      KernelBackwardReference.forward algebra s b t (i,j,PUnit.unit))=_
  rw [←Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro j _
  rw [value_gradient s scale b a g terminal saved,recompute_state s b a g terminal saved t (by omega),
    saved.residual t ht,saved.beta]
  unfold enhanced
  rw [enhanced_column s scale b a g terminal saved]
  simp only [gradient,Math.gradient,Math.pullback,Math.carry,enhancedColumn]
  have time : s.T-(t+1)=s.T-1-t := by omega
  rw [time]
  ring

theorem seed_gradient (s : Shape) (scale : ℝ) (b : KernelBackwardReference.Input s ℝ)
    (a : Fin s.BBV → Math.Input s.BK) (g : Nat → Fin s.BBV → ℝ) (terminal : Fin s.BBV → Math.Vec s.BK)
    (saved : Saved s b a g terminal) (j : Fin s.BBV) (i : Fin s.BK) :
    KernelBackwardReference.reverse algebra s scale b s.T (i,j,PUnit.unit)=
      Math.carry scale (a j) (fun n => g n j) (terminal j) s.T 0 i := by
  rw [reverse_state s scale b a g terminal saved]
  simp [Math.carry]

#axiomsClean value_gradient
#axiomsClean beta_gradient
#axiomsClean scalar_beta_gradient
#axiomsClean query_gradient
#axiomsClean key_gradient
#axiomsClean seed_gradient
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelReal
