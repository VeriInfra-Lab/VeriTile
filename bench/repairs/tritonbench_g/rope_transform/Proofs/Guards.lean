import bench.repairs.tritonbench_g.rope_transform.Proofs.HostContract

namespace VeriTile.Bench.Repairs.RopeTransform.Checked
open VeriTile Triton FP.Structural

theorem launch_admitted {α : Type} [Inhabited α] (r : Request) (call : Host.Preparation α r)
    (accepted : check r call=Bool.true) : ABI.Limits r ∧ ABI.PointerLimits r call := by
  simp only [check,Bool.and_eq_true] at accepted
  have h := accepted.2
  simp only [launchCheck,Bool.and_eq_true,decide_eq_true_eq] at h
  exact ⟨⟨h.1.1.1.1,h.1.1.1.2,h.1.1.2,h.1.2⟩,h.2⟩

theorem checked_indices {α : Type} [Inhabited α] (r : Request) (call : Host.Preparation α r)
    (accepted : check r call=Bool.true) (key second : Bool) (a h j : Nat)
    (ha : a<r.rows) (hh : h<r.paddedHeads key) (hj : j<r.width) :
    r.offset key a h (r.column second j)<2^31 ∧
      (call.output key).base+4*r.offset key a h (r.column second j)+4≤2^64 := by
  obtain ⟨limits,pointers⟩ := launch_admitted r call accepted
  exact ⟨(ABI.formed_output limits key second a h j ha hh hj).2,
    ABI.padded_complete_pointer r call limits pointers key second a h j ha hh hj⟩

theorem payload_fp32 (t : Host.Tensor) (h : payload t=Bool.true) :
    t.strided=Bool.true ∧ t.cuda=Bool.true ∧ t.contiguous=Bool.true ∧
    t.negative=Bool.false ∧ t.conjugate=Bool.false ∧ t.dtype="float32" ∧ t.elementBytes=4 := by
  simpa [payload,and_assoc] using h

#axiomsClean launch_admitted
#axiomsClean checked_indices
#axiomsClean payload_fp32
#specNonCircular check avoiding [Host.invoke,Grid.run,Grid.sourceRun,FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.RopeTransform.Checked
