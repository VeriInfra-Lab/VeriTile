import bench.tritonbench_g.rotary_transform.Proofs.Contract

namespace VeriTile.Bench.Repairs.RotaryTransform.Checked

theorem checks (c : Call) (initial : Physical.Bytes) (accepted : check c initial=true) :
    ((shapeCheck c.host.shape=true ∧ tensorCheck c=true) ∧
      metadataCheck (configuration c initial)=true) ∧ indexCheck (configuration c initial)=true := by
  simpa only [check,Bool.and_eq_true] using accepted

theorem config_valid (c : Call) (initial : Physical.Bytes) (accepted : check c initial=true) :
    (configuration c initial).Valid := by
  have hs := (checks c initial accepted).1.1.1
  have hm := (checks c initial accepted).1.2
  simp only [shapeCheck,Bool.and_eq_true,decide_eq_true_eq] at hs
  have hs := hs
  simp only [metadataCheck,Bool.and_eq_true,decide_eq_true_eq] at hm
  have mono : ∀ a b,a≤b → b≤c.host.shape.batches →
      (configuration c initial).metadata.cumulative a≤(configuration c initial).metadata.cumulative b := by
    intro a b hab hb
    induction b with
    | zero =>
      have ha : a=0 := by omega
      simp [ha]
    | succ b ih =>
      by_cases eq : a=b+1
      · simp [eq]
      · exact Nat.le_trans (ih (by omega) (by omega)) (hm.2 ⟨b,by change b<c.host.shape.batches; omega⟩).1
  refine ⟨hs.1,hs.2.1,hs.2.2.2,fun _ => hm.1.1,fun _ => mono,hm.1.2,?_⟩
  intro b hb
  exact (hm.2 ⟨b,hb⟩).2

theorem indices (c : Call) (initial : Physical.Bytes) (accepted : check c initial=true) :
    IndexABI (configuration c initial) := by
  have hi := (checks c initial accepted).2
  simp only [indexCheck,Bool.and_eq_true,decide_eq_true_eq] at hi
  exact ⟨hi.1.1,hi.1.2.1,hi.1.2.2.1,hi.1.2.2.2,
    fun b hb => (hi.2 ⟨b,hb⟩).1,fun b hb => (hi.2 ⟨b,hb⟩).2⟩

theorem admitted (c : Call) (initial : Physical.Bytes) (accepted : check c initial=true)
    (fresh : c.host.shape.inplace=false → c.host.freshIdentity≠c.host.inputIdentity) :
    Public.Admitted c.host c.views initial := by
  refine ⟨config_valid c initial accepted,indices c initial accepted,?_,?_,fresh⟩
  · have ht := ((checks c initial accepted).1.1.2)
    simp only [tensorCheck,Bool.and_eq_true,decide_eq_true_eq] at ht
    tauto
  · have ht := ((checks c initial accepted).1.1.2)
    simp only [tensorCheck,Bool.and_eq_true,decide_eq_true_eq] at ht
    have stride : c.host.abi.cumulativeStride=1 := by tauto
    omega

#axiomsClean config_valid
#axiomsClean indices
#axiomsClean admitted
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.RotaryTransform.Checked
