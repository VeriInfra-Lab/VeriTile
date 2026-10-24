import bench.repairs.tritonbench_g.sin_computation.Contract
import bench.repairs.tritonbench_g.sin_computation.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SinComputation.Contiguous

theorem contract_iff (x out : TensorMetadata) : InputContract x out ↔
    (x.layout = .strided ∧ out.layout = .strided) ∧
    (x.dtype = .floating .fp32 ∧ out.dtype = .floating .fp32) ∧
    (x.device.kind = "cuda" ∧ out.device.kind = "cuda") ∧
    x.device = out.device ∧ x.numel ≤ out.numel ∧
    (x.isContiguous = Bool.true ∧ out.isContiguous = Bool.true) ∧
    (x.isNeg = Bool.false ∧ out.isNeg = Bool.false) ∧ x.numel < 2^31 ∧
    (0 < x.numel → x.dataPtr.toNat % 4 = 0 ∧ out.dataPtr.toNat % 4 = 0) ∧
    (0 < x.numel → Compatible x out) := by
  constructor
  · intro h; exact ⟨h.layout,h.dtype,h.cuda,h.device,h.capacity,h.contiguous,h.noNeg,h.extent,h.alignment,h.overlap⟩
  · rintro ⟨a,b,c,d,e,f,g,h,i,j⟩; exact ⟨a,b,c,d,e,f,g,h,i,j⟩

theorem checkTensorInputs_iff (x out : TensorMetadata) :
    checkTensorInputs x out = .ok () ↔ InputContract x out := by
  rw [contract_iff]
  simp [checkTensorInputs, VeriTile.Torch.TensorMetadata.isCuda, ite_eq_iff]

theorem checkInputs_iff (x out : ArgumentMetadata) :
    checkInputs x out = .ok () ↔ AdmittedArguments x out := by
  cases x <;> cases out <;> simp [checkInputs,AdmittedArguments,checkTensorInputs_iff]

#axiomsClean checkInputs_iff
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SinComputation.Contiguous
