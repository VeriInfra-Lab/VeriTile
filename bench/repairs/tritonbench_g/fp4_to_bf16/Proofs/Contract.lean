import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.fp4_to_bf16.Contract
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.Fp4ToBf16.Checked
open VeriTile.Torch

theorem input_contract_iff (x : TensorMetadata) :
    InputContract x ↔ x.layout = .strided ∧ x.dtype = .other "uint8" ∧
      x.shape ≠ [] ∧ x.device.kind = "cuda" ∧ x.isContiguous = Bool.true ∧ x.isNeg = Bool.false := by
  constructor
  · intro h; exact ⟨h.layout,h.dtype,h.rank,h.cuda,h.contiguous,h.noNeg⟩
  · rintro ⟨hl,hd,hr,hc,hk,hn⟩; exact ⟨hl,hd,hr,hc,hk,hn⟩

theorem checkTensor_iff (x : TensorMetadata) :
    checkTensor x = .ok () ↔ InputContract x := by
  simp [checkTensor,input_contract_iff,TensorMetadata.isCuda,ite_eq_iff]

theorem checkInputs_iff (x : ArgumentMetadata) :
    checkInputs x = .ok () ↔ AdmittedArguments x := by
  cases x with
  | nonTensor => simp [checkInputs,AdmittedArguments]
  | tensor x => exact checkTensor_iff x

#axiomsClean checkInputs_iff
end VeriTile.Bench.Repairs.Fp4ToBf16.Checked
