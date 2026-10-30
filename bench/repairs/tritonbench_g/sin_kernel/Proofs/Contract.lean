import bench.repairs.tritonbench_g.sin_kernel.Contract
import bench.repairs.tritonbench_g.sin_kernel.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SinKernel

theorem contract_iff (x : TensorMetadata) : InputContract x ↔
    x.layout = .strided ∧ x.dtype = .floating .fp32 ∧ x.device.kind = "cuda" ∧
    x.isContiguous = Bool.true ∧ x.isNeg = Bool.false ∧ x.numel < 2^31 ∧
    (0 < x.numel → x.dataPtr.toNat % 4 = 0) := by
  constructor
  · intro h; exact ⟨h.layout,h.dtype,h.cuda,h.contiguous,h.noNeg,h.extent,h.alignment⟩
  · rintro ⟨a,b,c,d,e,f,g⟩; exact ⟨a,b,c,d,e,f,g⟩
theorem checkTensorInputs_iff (x : TensorMetadata) :
    checkTensorInputs x = .ok () ↔ InputContract x := by
  rw [contract_iff]
  simp [checkTensorInputs, VeriTile.Torch.TensorMetadata.isCuda, ite_eq_iff]
theorem checkInputs_iff (x : ArgumentMetadata) :
    checkInputs x = .ok () ↔ AdmittedArguments x := by
  cases x <;> simp [checkInputs, AdmittedArguments, checkTensorInputs_iff]


#axiomsClean checkInputs_iff
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SinKernel
