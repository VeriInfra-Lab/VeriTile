import bench.repairs.tritonbench_g.fused_recurrent_delta.Misc.HostContract
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.ABI
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Admission

theorem every_region (r : Region) : r∈regions := by cases r <;> simp [regions]

theorem checked_index (ok : shapeCheck c = Bool.true) : Index.Safe c.shape (ABI.flags c) := by
  simp only [shapeCheck,Bool.and_eq_true,decide_eq_true_eq] at ok
  refine ⟨ok.1.1,ok.1.2.1,ok.1.2.2.1,ok.1.2.2.2.1,?_⟩
  intro r
  exact of_decide_eq_true (List.all_eq_true.mp ok.2 r (every_region r))

theorem checked_parts (ok : check c device tensors outputIsBool scaleAccepted = Bool.true) :
    outputIsBool=Bool.true ∧ scaleAccepted=Bool.true ∧ shapeCheck c=Bool.true ∧
    flagsCheck c=Bool.true ∧ c.scale.launch.toRat?.isSome=Bool.true ∧ tensorChecks c device tensors=Bool.true := by
  simpa only [check,Bool.and_eq_true,and_assoc] using ok

theorem checked_admitted (ok : check c device tensors outputIsBool scaleAccepted = Bool.true) : ABI.Admitted c := by
  have parts := checked_parts ok
  have flags : (c.defaultBeta=Bool.true → c.vectorBeta=Bool.false) ∧
      (c.terminalCotangent=Bool.true → c.outputFinal=Bool.true) := of_decide_eq_true parts.2.2.2.1
  refine ⟨checked_index parts.2.2.1,flags.1,flags.2,?_⟩
  exact Option.isSome_iff_exists.mp parts.2.2.2.2.1

theorem checked_tensor (ok : check c device tensors outputIsBool scaleAccepted = Bool.true)
    (r : Region) (d : Descriptor) (member : (r,d)∈tensors) :
    descriptorCheck d (ABI.tensorMeta c.shape (ABI.flags c) r).shape device=Bool.true ∧
    windowCheck c r d=Bool.true := by
  have ht := (checked_parts ok).2.2.2.2.2
  simp only [tensorChecks,Bool.and_eq_true] at ht
  have hx := List.all_eq_true.mp ht.1.1 (r,d) member
  simpa only [Bool.and_eq_true] using hx

theorem checked_active_bytes (ok : check c device tensors outputIsBool scaleAccepted = Bool.true)
    (r : Region) (d : Descriptor) (member : (r,d)∈tensors) (a : Addr)
    (ha : Geometry.Valid c.shape (ABI.flags c) a) (hr : a.region=r) (byte : Fin 4) :
    d.base+4*Geometry.offset c.shape (ABI.flags c) a+byte.val<2^64 ∧
    (d.base+4*Geometry.offset c.shape (ABI.flags c) a)%4=0 := by
  have hw := (checked_tensor ok r d member).2
  simp only [windowCheck,decide_eq_true_eq] at hw
  have hb := Geometry.offset_bound c.shape (ABI.flags c) a ha
  rw [hr] at hb
  have hs : Geometry.capacity c.shape (ABI.flags c) r≤span c.shape (ABI.flags c) r := by unfold span; omega
  have hj := byte.isLt
  exact ⟨by omega,by simp [Nat.add_mod,hw.1]⟩

theorem checked_backward (ok : backwardCheck alreadyUsed higherOrder = Bool.true) :
    alreadyUsed=Bool.false ∧ higherOrder=Bool.false := by
  simpa [backwardCheck] using ok

theorem checked_separation (ok : check c device tensors outputIsBool scaleAccepted=Bool.true)
    (r t : Region) (d e : Descriptor) (hd : (r,d)∈tensors) (he : (t,e)∈tensors) (different : r≠t) :
    d.base+4*Geometry.capacity c.shape (ABI.flags c) r≤e.base ∨
    e.base+4*Geometry.capacity c.shape (ABI.flags c) t≤d.base := by
  have ht := (checked_parts ok).2.2.2.2.2
  simp only [tensorChecks,Bool.and_eq_true] at ht
  have pair := List.all_eq_true.mp (List.all_eq_true.mp ht.1.2 (r,d) hd) (t,e) he
  have pair' : r=t ∨ d.base+4*Geometry.capacity c.shape (ABI.flags c) r≤e.base ∨
      e.base+4*Geometry.capacity c.shape (ABI.flags c) t≤d.base := of_decide_eq_true pair
  exact pair'.resolve_left different

#axiomsClean checked_index
#axiomsClean checked_admitted
#axiomsClean checked_tensor
#axiomsClean checked_active_bytes
#axiomsClean checked_backward
#axiomsClean checked_separation
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Admission
