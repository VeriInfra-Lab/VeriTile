import bench.repairs.tritonbench_g.layernorm_fwd_triton.Contract
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.Contract
open VeriTile.Torch
noncomputable section

theorem checkTensor_iff (m : TensorMetadata) : checkTensor m = .ok () ↔ TensorContract m := by
  have unpack : TensorContract m ↔ m.layout = .strided ∧ m.dtype = .floating .fp32 ∧
      m.device.kind = "cuda" ∧ m.isNeg = Bool.false ∧ (0 < m.numel → m.dataPtr.toNat % 4 = 0) :=
    ⟨fun h => ⟨h.layout,h.dtype,h.cuda,h.noNeg,h.alignment⟩,fun ⟨a,b,c,d,e⟩ => ⟨a,b,c,d,e⟩⟩
  rw [unpack]
  simp [checkTensor, ite_eq_iff, and_assoc]

theorem checkInputs_iff (x w : TensorMetadata) (eps : Epsilon) :
    checkInputs x w eps = .ok () ↔ InputContract x w eps := by
  have unpack : InputContract x w eps ↔ TensorContract x ∧ TensorContract w ∧
      x.shape.length = 3 ∧ w.shape.length = 2 ∧ w.shape = x.shape.drop 1 ∧
      w.device = x.device ∧ eps.numeric = Bool.true ∧ eps.finite = Bool.true ∧ 0 < eps.value :=
    ⟨fun h => ⟨h.input,h.weight,h.rank,h.weightRank,h.weightShape,h.device,h.epsilon⟩,
      fun ⟨a,b,c,d,e,f,g,h,i⟩ => ⟨a,b,c,d,e,f,⟨g,h,i⟩⟩⟩
  rw [unpack]
  simp only [← checkTensor_iff]
  cases hx : checkTensor x <;> cases hw : checkTensor w <;>
    simp [checkInputs,hx,hw,ite_eq_iff,and_assoc]

theorem checkArguments_iff (x w : ArgumentMetadata) (eps : Epsilon) :
    checkArguments x w eps = .ok () ↔ AdmittedArguments x w eps := by
  cases x <;> cases w <;> simp [checkArguments,AdmittedArguments,checkInputs_iff]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.Contract
