import bench.tritonbench_g.masked_add_cuda.Contract
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.MaskedAddCuda

theorem checkTensorInputs_iff (g p m : TensorMetadata) (a : AlphaArgument) :
    checkTensorInputs g p m a = .ok () ↔ InputContract g p m a := by
  classical
  by_cases empty : g.numel = 0
  · simp [checkTensorInputs, InputContract, empty,
      VeriTile.Torch.TensorMetadata.isCuda, ite_eq_iff, and_assoc]
    tauto
  · have positive : 0 < g.numel := Nat.pos_of_ne_zero empty
    simp [checkTensorInputs, InputContract, empty, positive,
      VeriTile.Torch.TensorMetadata.isCuda, ite_eq_iff, and_assoc]
    tauto

theorem checkInputs_iff (g p m : ArgumentMetadata) (a : AlphaArgument) :
    checkInputs g p m a = .ok () ↔ AdmittedArguments g p m a := by
  cases g <;> cases p <;> cases m <;>
    simp [checkInputs, AdmittedArguments, checkTensorInputs_iff]

theorem selected_mask_dtype (g p m : TensorMetadata) (a : AlphaArgument)
    (h : InputContract g p m a) : m.dtype = maskDType (selectedMask m) := by
  rcases h.2.1.2.2 with h32 | h64
  · simp [selectedMask, maskDType, h32]
  · simp [selectedMask, maskDType, h64]

#axiomsClean checkTensorInputs_iff
#axiomsClean checkInputs_iff
#axiomsClean selected_mask_dtype
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.MaskedAddCuda
