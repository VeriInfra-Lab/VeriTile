import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelContract

namespace VeriTile.Bench.Repairs.TritonLinearActivation.KernelPointer
open VeriTile.Triton KernelContract
noncomputable section
set_option maxHeartbeats 3000000

/-- The observed kernel Tensor pointer is its actual backing base when the
source's zero-offset preparation applies. This follows from ordinary API laws. -/
theorem actual_pointer (mode : Mode) (t : ActualTensor mode)
    (offset : storageOffset mode t=0) (nonempty : 0<(metadata mode t).numel) :
    (metadata mode t).dataPtr=BitVec.ofNat 64 (storageBase mode t) := by
  cases mode
  · change t.val.storageOffset=0 at offset
    simpa [metadata,storageBase,VeriTile.Torch.FP32.Storage.address,offset] using t.property.pointer nonempty
  · change t.val.storageOffset=0 at offset
    simpa [metadata,storageBase,offset] using t.property.pointer nonempty

theorem actual_address (mode : Mode) (t : ActualTensor mode)
    (contiguous : (metadata mode t).isContiguous=Bool.true)
    (offset : storageOffset mode t=0) (i : Nat) (valid : i<(metadata mode t).numel) :
    storageBase mode t+elementBytes mode*physicalIndex mode t i=
      (metadata mode t).dataPtr.toNat+elementBytes mode*i := by
  have cap := capacity mode t contiguous offset i valid
  have positive : 0<elementBytes mode := by cases mode <;> decide
  have span : storageBase mode t+elementBytes mode*storageElements mode t≤2^64 := by
    cases mode
    · exact t.val.storage.spanFits
    · exact t.val.spanFits
  have baseBound : storageBase mode t<2^64 := by nlinarith
  rw [physical_contiguous mode t contiguous offset i valid,
    actual_pointer mode t offset (by omega),BitVec.toNat_ofNat,Nat.mod_eq_of_lt baseBound]

#axiomsClean actual_pointer
#axiomsClean actual_address
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonLinearActivation.KernelPointer
