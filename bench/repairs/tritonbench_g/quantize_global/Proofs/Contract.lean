import Mathlib
import bench.repairs.tritonbench_g.quantize_global.Contract
import VeriTile.Meta.StatementAudit
namespace QuantizeGlobalRepair.Direct

theorem contract_iff (x : TensorMetadata) (allFinite : Bool) : InputContract x allFinite ↔
    x.layout=.strided ∧ x.dtype=.floating .fp32 ∧ x.device.kind="cuda" ∧
    x.isContiguous=Bool.true ∧ x.isNeg=Bool.false ∧ x.numel<2^63 ∧
    (0<x.numel → x.dataPtr.toNat%4=0) ∧ (x.numel=0 ∨ allFinite=Bool.true) := by
  constructor
  · intro h;exact ⟨h.layout,h.dtype,h.cuda,h.contiguous,h.noNeg,h.extent,h.alignment,h.finite⟩
  · rintro ⟨a,b,c,d,e,f,g,h⟩;exact ⟨a,b,c,d,e,f,g,h⟩
theorem checkTensorInputs_iff (x : TensorMetadata) (finite : Bool) :
    checkTensorInputs x finite=.ok () ↔ InputContract x finite := by
  rw [contract_iff]
  simp [checkTensorInputs,VeriTile.Torch.TensorMetadata.isCuda,ite_eq_iff,and_assoc]
  tauto
theorem checkInputs_iff (x : ArgumentMetadata) (finite : Bool) :
    checkInputs x finite=.ok () ↔ AdmittedArguments x finite := by
  cases x <;> simp [checkInputs,AdmittedArguments,checkTensorInputs_iff]
#axiomsClean checkInputs_iff
#guard_msgs (drop info) in
#auditModuleAxioms
end QuantizeGlobalRepair.Direct
