import bench.tritonbench_g.fused_recurrent_delta.IO
import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelRealMemory
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.LogicalMemory
open VeriTile.Triton FP.Structural KernelRegisters KernelBackwardRegisters Interface
noncomputable section
variable {α : Type} [Inhabited α]

theorem enhanced_time (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (u : State α) (t : Nat) (ht : t<s.T) :
    KernelReverseMemory.enhanced M s f scale bh iv u (s.T-1-t)=
      enhanced M s scale (KernelBackwardReference.input M s f bh iv u) t := by
  have time : s.T-1-(s.T-1-t)=t := by omega
  simp only [KernelReverseMemory.enhanced,KernelReverseMemory.carry,enhanced,time]

theorem key_reload (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (u : State α) (t : Nat) (ht : t<s.T) (i : Fin s.BK) :
    KernelReference.read M (KernelBackward.start M s f scale bh iv u) "dk"
      (keyAddress s (plane s bh iv) t i.val) (decide (i.val<s.K))=
      KernelReference.cast M (if i.val<s.K then KernelReference.cast M
        (KernelBackwardReference.dk M s (KernelBackwardReference.input M s f bh iv u) t
          (enhanced M s scale (KernelBackwardReference.input M s f bh iv u) t) i)
        else KernelReference.zero M) := by
  by_cases hi : i.val<s.K
  · have time : s.T-1-(s.T-1-t)=t := by omega
    have stored := KernelBackwardResults.reverse_key M s f scale bh iv u s.T (s.T-1-t) (by omega) le_rfl i hi
    rw [time,enhanced_time M s f scale bh iv u t ht] at stored
    simp only [KernelReference.read,hi,decide_true,ite_true]
    rw [KernelBackwardFacts.start_other M s f scale bh iv u "dk" _ (by decide),stored]
    rfl
  · simp [KernelReference.read,hi]

theorem value_reload (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (u : State α) (t : Nat) (ht : t<s.T) (j : Fin s.BBV) :
    KernelReference.read M (KernelBackward.start M s f scale bh iv u) "dv"
      (valAddress s bh iv s.BBV t j.val) (decide (iv*s.BBV+j.val<s.V))=
      KernelReference.cast M (if iv*s.BBV+j.val<s.V then KernelReference.cast M
        (KernelBackwardReference.dv M s (KernelBackwardReference.input M s f bh iv u) t
          (enhanced M s scale (KernelBackwardReference.input M s f bh iv u) t) j)
        else KernelReference.zero M) := by
  by_cases hj : iv*s.BBV+j.val<s.V
  · have time : s.T-1-(s.T-1-t)=t := by omega
    have stored := KernelBackwardResults.reverse_value M s f scale bh iv u s.T (s.T-1-t) (by omega) le_rfl j hj
    rw [time,enhanced_time M s f scale bh iv u t ht] at stored
    simp only [KernelReference.read,hj,decide_true,ite_true]
    rw [KernelBackwardFacts.start_other M s f scale bh iv u "dv" _ (by decide),stored]
    rfl
  · simp [KernelReference.read,hj]

theorem query (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (u : State α) (t : Nat) :
    KernelRecomputeReferenceMemory.query M s f scale bh iv (KernelBackward.start M s f scale bh iv u) t=
      backwardQuery M s scale (KernelBackwardReference.input M s f bh iv u) t := by
  unfold KernelRecomputeReferenceMemory.query KernelRecomputeReferenceMemory.carry backwardQuery
  rw [KernelReal.input_start]

theorem key (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (u : State α) (t : Nat) (ht : t<s.T) :
    KernelRecomputeReferenceMemory.corrected M s f bh iv (KernelBackward.start M s f scale bh iv u) t=
      backwardKey M s scale iv (KernelBackwardReference.input M s f bh iv u) t := by
  unfold KernelRecomputeReferenceMemory.corrected KernelRecomputeReferenceMemory.carry
  rw [KernelReal.input_start]
  unfold KernelRecomputeValues.dk backwardKey
  simp_rw [key_reload M s f scale bh iv u t ht,value_reload M s f scale bh iv u t ht]

#axiomsClean enhanced_time
#axiomsClean key_reload
#axiomsClean value_reload
#axiomsClean query
#axiomsClean key
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.LogicalMemory
