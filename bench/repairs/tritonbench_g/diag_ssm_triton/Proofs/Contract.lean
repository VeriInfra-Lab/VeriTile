import bench.repairs.tritonbench_g.diag_ssm_triton.Contract
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic

namespace VeriTile.Bench.Repairs.DiagSsmTriton.Contract

theorem tensor_contract_iff (c : Bool) (m : Metadata) :
    TensorContract c m ↔ m.layout = .strided ∧ SupportedDType m ∧ m.device.kind = "cuda" ∧
      (c = Bool.true → m.isContiguous = Bool.true) ∧
      (m.isNeg = Bool.false ∧ m.isConj = Bool.false) ∧
      (0 < m.numel → m.dataPtr.toNat % 4 = 0) := by
  exact ⟨fun h => ⟨h.layout, h.dtype, h.cuda, h.dense, h.resolved, h.alignment⟩,
    fun ⟨a,b,c,d,e,f⟩ => ⟨a,b,c,d,e,f⟩⟩

theorem checkTensor_iff (c : Bool) (m : Metadata) :
    checkTensor c m = .ok () ↔ TensorContract c m := by
  rw [tensor_contract_iff]
  cases c <;> simp [checkTensor, SupportedDType, ite_eq_iff] <;> tauto

theorem input_contract_iff (s x transition : Metadata) :
    InputContract s x transition ↔
      TensorContract Bool.true s ∧ TensorContract Bool.true x ∧ TensorContract Bool.true transition ∧
      (s.dtype = x.dtype ∧ x.dtype = transition.dtype) ∧
      (s.device = x.device ∧ x.device = transition.device) ∧ x.shape.length = 3 ∧
      s.shape = x.shape.drop 1 ∧ transition.shape = x.shape.drop 2 := by
  exact ⟨fun h => ⟨h.initial,h.input,h.parameter,h.dtype,h.device,h.rank,h.initialShape,h.parameterShape⟩,
    fun ⟨a,b,c,d,e,f,g,h⟩ => ⟨a,b,c,d,e,f,g,h⟩⟩

theorem checkInputs_iff (s x transition : Metadata) :
    checkInputs s x transition = .ok () ↔ InputContract s x transition := by
  rw [input_contract_iff]
  simp only [← checkTensor_iff]
  cases hs : checkTensor Bool.true s <;> cases hx : checkTensor Bool.true x <;>
    cases hl : checkTensor Bool.true transition <;>
    simp [checkInputs, hs, hx, hl, ite_eq_iff]

theorem checkGradient_iff (gy y : Metadata) :
    checkGradient gy y = .ok () ↔ GradientContract gy y := by
  have contract : GradientContract gy y ↔ TensorContract Bool.false gy ∧
      gy.dtype = y.dtype ∧ gy.device = y.device ∧ gy.shape = y.shape :=
    ⟨fun h => ⟨h.tensor,h.dtype,h.device,h.shape⟩, fun ⟨a,b,c,d⟩ => ⟨a,b,c,d⟩⟩
  rw [contract]
  simp only [← checkTensor_iff]
  cases hg : checkTensor Bool.false gy <;> simp [checkGradient, hg, ite_eq_iff, and_assoc]

theorem checkArguments_iff (s x transition : Argument) :
    checkArguments s x transition = .ok () ↔ AdmittedArguments s x transition := by
  cases s <;> cases x <;> cases transition <;>
    simp [checkArguments, AdmittedArguments, checkInputs_iff]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.Contract
