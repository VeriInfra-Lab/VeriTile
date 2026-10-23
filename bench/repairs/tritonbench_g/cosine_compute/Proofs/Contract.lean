import bench.repairs.tritonbench_g.cosine_compute.Contract
import bench.repairs.tritonbench_g.cosine_compute.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.CosineCompute

theorem contract_iff (x : TensorMetadata) : InputContract x ↔
    x.layout = .strided ∧ x.dtype = .floating .fp32 ∧ x.device.kind = "cuda" ∧
    x.isContiguous = Bool.true ∧ x.isNeg = Bool.false ∧ x.numel < 2^31 ∧
    (0 < x.numel → x.dataPtr.toNat % 4 = 0) := by
  constructor
  · intro h; exact ⟨h.layout,h.dtype,h.cuda,h.contiguous,h.noNeg,h.extent,h.alignment⟩
  · rintro ⟨hl,hd,hc,ht,hn,he,ha⟩; exact ⟨hl,hd,hc,ht,hn,he,ha⟩

theorem checkTensorInput_iff (x : TensorMetadata) :
    checkTensorInput x = .ok () ↔ InputContract x := by
  rw [contract_iff]
  simp [checkTensorInput, VeriTile.Torch.TensorMetadata.isCuda, ite_eq_iff]

theorem checkInput_iff (x : ArgumentMetadata) :
    checkInput x = .ok () ↔ AdmittedArgument x := by
  cases x <;>  simp [checkInput, AdmittedArgument, checkTensorInput_iff]


#axiomsClean checkTensorInput_iff
#axiomsClean checkInput_iff
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.CosineCompute
