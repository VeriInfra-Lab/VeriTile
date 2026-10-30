import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelRealGradients
import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelBackwardFacts
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelReal
open VeriTile.Triton FP.Structural KernelRegisters KernelBackwardRegisters
open scoped BigOperators
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 14000000

theorem start_readonly {α : Type} [Inhabited α] (M : Algebra α) (s : Shape) (f : Flags)
    (scale : ℝ) (bh iv : Nat) (u : State α) (r : RegionName) (off : Nat)
    (hk : r≠"dk") (hv : r≠"dv") (hb : r≠"dbeta") (hh : r≠"dh0") :
    (KernelBackward.start M s f scale bh iv u).mem r off=u.mem r off := by
  rw [KernelBackwardFacts.start_other M s f scale bh iv u r off hh]
  exact KernelReverseMemory.readonly M s f scale bh iv u s.T r off hk hv hb
theorem input_start {α : Type} [Inhabited α] (M : Algebra α) (s : Shape) (f : Flags)
    (scale : ℝ) (bh iv : Nat) (u : State α) :
    KernelBackwardReference.input M s f bh iv (KernelBackward.start M s f scale bh iv u)=
      KernelBackwardReference.input M s f bh iv u := by
  simp only [KernelBackwardReference.input,KernelReference.read,
    start_readonly M s f scale bh iv u "q" _ (by decide) (by decide) (by decide) (by decide),
    start_readonly M s f scale bh iv u "k" _ (by decide) (by decide) (by decide) (by decide),
    start_readonly M s f scale bh iv u "v" _ (by decide) (by decide) (by decide) (by decide),
    start_readonly M s f scale bh iv u "beta" _ (by decide) (by decide) (by decide) (by decide),
    start_readonly M s f scale bh iv u "do" _ (by decide) (by decide) (by decide) (by decide),
    start_readonly M s f scale bh iv u "h0" _ (by decide) (by decide) (by decide) (by decide),
    start_readonly M s f scale bh iv u "dht" _ (by decide) (by decide) (by decide) (by decide)]

theorem enhanced_time (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (u : State ℝ)
    (t : Nat) (ht : t<s.T) :
    KernelReverseMemory.enhanced algebra s f scale bh iv u (s.T-1-t)=
      enhanced s scale (KernelBackwardReference.input algebra s f bh iv u) t := by
  have time : s.T-1-(s.T-1-t)=t := by omega
  simp only [KernelReverseMemory.enhanced,KernelReverseMemory.carry,enhanced,time]

theorem reloaded_key (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (u : State ℝ)
    (t : Nat) (ht : t<s.T) (i : Fin s.BK) (hi : i.val<s.K) :
    KernelReference.read algebra (KernelBackward.start algebra s f scale bh iv u) "dk"
      (keyAddress s (plane s bh iv) t i.val) (decide (i.val<s.K))=
      KernelBackwardReference.dk algebra s (KernelBackwardReference.input algebra s f bh iv u) t
        (enhanced s scale (KernelBackwardReference.input algebra s f bh iv u) t) i := by
  have time : s.T-1-(s.T-1-t)=t := by omega
  have stored := KernelBackwardResults.reverse_key algebra s f scale bh iv u s.T (s.T-1-t) (by omega) le_rfl i hi
  rw [time,enhanced_time s f scale bh iv u t ht] at stored
  simp only [KernelReference.read,hi,decide_true,ite_true]
  rw [KernelBackwardFacts.start_other algebra s f scale bh iv u "dk" _ (by decide),stored]
  rfl
theorem reloaded_value (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (u : State ℝ)
    (t : Nat) (ht : t<s.T) (j : Fin s.BBV) (hj : iv*s.BBV+j.val<s.V) :
    KernelReference.read algebra (KernelBackward.start algebra s f scale bh iv u) "dv"
      (valAddress s bh iv s.BBV t j.val) (decide (iv*s.BBV+j.val<s.V))=
      KernelBackwardReference.dv algebra s (KernelBackwardReference.input algebra s f bh iv u) t
        (enhanced s scale (KernelBackwardReference.input algebra s f bh iv u) t) j := by
  have time : s.T-1-(s.T-1-t)=t := by omega
  have stored := KernelBackwardResults.reverse_value algebra s f scale bh iv u s.T (s.T-1-t) (by omega) le_rfl j hj
  rw [time,enhanced_time s f scale bh iv u t ht] at stored
  simp only [KernelReference.read,hj,decide_true,ite_true]
  rw [KernelBackwardFacts.start_other algebra s f scale bh iv u "dv" _ (by decide),stored]
  rfl

theorem recompute_padding (s : Shape) (f : Flags) (bh iv : Nat) (u : State ℝ)
    (n : Nat) (i : Fin s.BK) (j : Fin s.BBV) (padding : ¬iv*s.BBV+j.val<s.V) :
    KernelBackwardReference.forward algebra s (KernelBackwardReference.input algebra s f bh iv u) n (i,j,PUnit.unit)=0 := by
  induction n with
  | zero =>
    simp [KernelBackwardReference.forward,KernelBackwardReference.input,KernelReference.read,
      KernelReference.cast,KernelReference.zero,KernelReference.add,algebra,padding]
  | succ n ih =>
    change KernelBackwardReference.forward algebra s (KernelBackwardReference.input algebra s f bh iv u) n (i,j,PUnit.unit)+
      (KernelBackwardReference.input algebra s f bh iv u).key n i*
        ((KernelBackwardReference.input algebra s f bh iv u).value n j*
          (KernelBackwardReference.input algebra s f bh iv u).beta n j)=0
    rw [ih]
    simp [KernelBackwardReference.input,KernelReference.read,KernelReference.cast,KernelReference.zero,algebra,padding]

theorem query_memory (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (u : State ℝ) (t : Nat) :
    KernelRecomputeReferenceMemory.query algebra s f scale bh iv (KernelBackward.start algebra s f scale bh iv u) t=
      KernelBackwardReference.dq algebra s scale (KernelBackwardReference.input algebra s f bh iv u) t
        (KernelBackwardReference.forward algebra s (KernelBackwardReference.input algebra s f bh iv u) (t+1)) := by
  unfold KernelRecomputeReferenceMemory.query KernelRecomputeReferenceMemory.carry
  rw [input_start]
theorem corrected_memory (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) (u : State ℝ)
    (t : Nat) (ht : t<s.T) (i : Fin s.BK) (hi : i.val<s.K) :
    KernelRecomputeReferenceMemory.corrected algebra s f bh iv (KernelBackward.start algebra s f scale bh iv u) t i=
      KernelBackwardReference.correction algebra s
        (KernelBackwardReference.forward algebra s (KernelBackwardReference.input algebra s f bh iv u) t)
        (KernelBackwardReference.dk algebra s (KernelBackwardReference.input algebra s f bh iv u) t
          (enhanced s scale (KernelBackwardReference.input algebra s f bh iv u) t))
        (KernelBackwardReference.dv algebra s (KernelBackwardReference.input algebra s f bh iv u) t
          (enhanced s scale (KernelBackwardReference.input algebra s f bh iv u) t)) i := by
  unfold KernelRecomputeReferenceMemory.corrected KernelRecomputeReferenceMemory.carry
  rw [input_start]
  unfold KernelRecomputeValues.dk KernelBackwardReference.correction
  rw [value_sum,value_sum]
  simp only [KernelReference.sub,KernelReference.mul]
  change KernelReference.read algebra (KernelBackward.start algebra s f scale bh iv u) "dk"
    (keyAddress s (plane s bh iv) t i.val) (decide (i.val<s.K))-(∑ j : Fin s.BBV,
      KernelReference.read algebra (KernelBackward.start algebra s f scale bh iv u) "dv"
        (valAddress s bh iv s.BBV t j.val) (decide (iv*s.BBV+j.val<s.V))*
      KernelBackwardReference.forward algebra s (KernelBackwardReference.input algebra s f bh iv u) t (i,j,PUnit.unit))=_
  rw [reloaded_key s f scale bh iv u t ht i hi]
  congr 1
  apply Finset.sum_congr rfl
  intro j _
  by_cases active : iv*s.BBV+j.val<s.V
  · rw [reloaded_value s f scale bh iv u t ht j active]
    rfl
  · rw [recompute_padding s f bh iv u t i j active]
    simp [algebra]

#axiomsClean input_start
#axiomsClean reloaded_key
#axiomsClean reloaded_value
#axiomsClean recompute_padding
#axiomsClean query_memory
#axiomsClean corrected_memory
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelReal
