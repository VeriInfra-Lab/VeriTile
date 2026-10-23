import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.IOSaved
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.IOResult
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.IORealBackward
open VeriTile.Triton FP.Structural KernelRegisters KernelBackwardRegisters Interface
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 4000000

theorem padding (s : Shape) (f : Flags) (iv : Nat) (vals : BackwardInputs ℝ s)
    (n : Nat) (i : Fin s.BK) (j : Fin s.BBV) (inactive : ¬iv*s.BBV+j.val<s.V) :
    KernelBackwardReference.forward KernelReal.algebra s (backwardData KernelReal.algebra s f iv vals) n (i,j,())=0 := by
  induction n with
  | zero =>
    simp [KernelBackwardReference.forward,backwardData,KernelReference.zero,KernelReference.cast,
      KernelReference.add,KernelReal.algebra,inactive]
  | succ n ih =>
    change KernelBackwardReference.forward KernelReal.algebra s (backwardData KernelReal.algebra s f iv vals) n (i,j,())+
      (backwardData KernelReal.algebra s f iv vals).key n i*
        ((backwardData KernelReal.algebra s f iv vals).value n j*
          (backwardData KernelReal.algebra s f iv vals).beta n j)=0
    rw [ih]
    by_cases ht : n<s.T <;>
      simp [backwardData,ht,inactive,KernelReference.zero,KernelReference.cast,KernelReal.algebra]

theorem key_unmasked (s : Shape) (f : Flags) (scale : ℝ) (iv : Nat) (vals : BackwardInputs ℝ s)
    (t : Nat) (i : Fin s.BK) (active : i.val<s.K) :
    backwardKey KernelReal.algebra s scale iv (backwardData KernelReal.algebra s f iv vals) t i=
      KernelBackwardReference.correction KernelReal.algebra s
        (KernelBackwardReference.forward KernelReal.algebra s (backwardData KernelReal.algebra s f iv vals) t)
        (KernelBackwardReference.dk KernelReal.algebra s (backwardData KernelReal.algebra s f iv vals) t
          (enhanced KernelReal.algebra s scale (backwardData KernelReal.algebra s f iv vals) t))
        (KernelBackwardReference.dv KernelReal.algebra s (backwardData KernelReal.algebra s f iv vals) t
          (enhanced KernelReal.algebra s scale (backwardData KernelReal.algebra s f iv vals) t)) i := by
  unfold backwardKey KernelBackwardReference.correction
  rw [KernelReal.value_sum,KernelReal.value_sum]
  simp only [KernelReference.sub,KernelReference.mul,KernelReference.cast,KernelReference.zero,
    KernelReal.algebra,if_pos active]
  congr 1
  apply Finset.sum_congr rfl
  intro j _
  by_cases hj : iv*s.BBV+j.val<s.V
  · simp only [if_pos hj]
  · rw [if_neg hj]
    have padded := padding s f iv vals t i j hj
    simp only [KernelReal.algebra] at padded
    rw [padded]
    simp

theorem formula (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (a : Fin s.BBV → Math.Input s.BK) (g : Nat → Fin s.BBV → ℝ)
    (terminal : Fin s.BBV → Math.Vec s.BK) (vals : BackwardInputs ℝ s)
    (ready : SavedInputs s f iv a g terminal vals) :
    ∀ port j,(backwardIO KernelReal.algebra s f scale bh iv).outputMask port vals j →
      backwardReference KernelReal.algebra s f scale iv vals port j=
        backwardFormula s scale a g terminal vals port j := by
  let b := backwardData KernelReal.algebra s f iv vals
  have saved := IOSaved.cutoff_saved s f iv a g terminal vals ready
  rintro ⟨port,hp⟩
  match port with
  | 0 =>
    intro j _
    have value := KernelReal.query_gradient s scale b (fun v => MathAgreement.cutoff s.T (a v))
      (IOSaved.cutIncoming s g) terminal saved j.1.val j.1.isLt j.2.1
    simp_rw [IOSaved.gradient s scale a g terminal j.1.val j.1.isLt] at value
    exact value
  | 1 =>
    intro j active
    change backwardKey KernelReal.algebra s scale iv (backwardData KernelReal.algebra s f iv vals) j.1.val j.2.1=_
    rw [key_unmasked s f scale iv vals j.1.val j.2.1 active]
    have value := KernelReal.key_gradient s scale b (fun v => MathAgreement.cutoff s.T (a v))
      (IOSaved.cutIncoming s g) terminal saved j.1.val j.1.isLt j.2.1
    simp_rw [IOSaved.gradient s scale a g terminal j.1.val j.1.isLt] at value
    exact value
  | 2 =>
    intro j _
    have value := KernelReal.value_gradient s scale b (fun v => MathAgreement.cutoff s.T (a v))
      (IOSaved.cutIncoming s g) terminal saved j.1.val j.2.1
    rw [IOSaved.gradient s scale a g terminal j.1.val j.1.isLt] at value
    exact value
  | 3 =>
    intro j _
    have value := KernelReal.beta_gradient s scale b (fun v => MathAgreement.cutoff s.T (a v))
      (IOSaved.cutIncoming s g) terminal saved j.1.val j.1.isLt j.2.1
    rw [IOSaved.gradient s scale a g terminal j.1.val j.1.isLt] at value
    exact value
  | 4 =>
    intro j _
    have value := KernelReal.scalar_beta_gradient s scale b (fun v => MathAgreement.cutoff s.T (a v))
      (IOSaved.cutIncoming s g) terminal saved j.1.val j.1.isLt
    simp_rw [IOSaved.gradient s scale a g terminal j.1.val j.1.isLt] at value
    exact value
  | 5 =>
    intro j _
    have value := KernelReal.seed_gradient s scale b (fun v => MathAgreement.cutoff s.T (a v))
      (IOSaved.cutIncoming s g) terminal saved j.2.1 j.1
    rw [IOSaved.seed] at value
    cases j with | mk i j => cases j with | mk j tail => cases tail; exact value
  | n+6 => simp [backwardIO,backwardOutputs] at hp

theorem complete (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (a : Fin s.BBV → Math.Input s.BK) (g : Nat → Fin s.BBV → ℝ)
    (terminal : Fin s.BBV → Math.Vec s.BK) :
    (backwardIO KernelReal.algebra s f scale bh iv).ImplementsOn (SavedInputs s f iv a g terminal)
      (backwardFormula s scale a g terminal) := by
  intro vals ready bounds hb initial hin
  obtain ⟨safe,final,ran,output,frame⟩ := IOResult.backward KernelReal.algebra s f scale bh iv vals trivial bounds hb initial hin
  refine ⟨safe,final,ran,?_,frame⟩
  intro port j active
  rw [←formula s f scale bh iv a g terminal vals ready port j active]
  exact output port j active

#axiomsClean padding
#axiomsClean key_unmasked
#axiomsClean formula
#axiomsClean complete
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.IORealBackward
