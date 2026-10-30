import bench.tritonbench_g.fast_layernorm.Contract
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic

namespace VeriTile.Bench.Repairs.FastLayernorm.Contract
open VeriTile.Torch
noncomputable section

theorem checkTensor_iff (m : TensorMetadata) (c : Bool) : checkTensor m c=.ok () ↔ TensorContract m c := by
  have unpack : TensorContract m c ↔ m.layout=.strided ∧ m.dtype=.floating .fp32 ∧
      m.device.kind="cuda" ∧ (c=Bool.true → m.isContiguous=Bool.true) ∧
      m.isNeg=Bool.false ∧ (0<m.numel → m.dataPtr.toNat%4=0) :=
    ⟨fun h => ⟨h.layout,h.dtype,h.cuda,h.dense,h.noNeg,h.alignment⟩,
      fun ⟨a,b,c,d,e,f⟩ => ⟨a,b,c,d,e,f⟩⟩
  rw [unpack]
  cases c <;> simp [checkTensor,ite_eq_iff,and_assoc]

theorem checkOptional_iff (bias : Option TensorMetadata) :
    checkOptional bias=.ok () ↔ ∀ b,bias=some b → TensorContract b Bool.true := by
  cases bias <;> simp [checkOptional,checkTensor_iff]

theorem checkInputs_iff (x w : TensorMetadata) (bias : Option TensorMetadata) (eps : Epsilon) :
    checkInputs x w bias eps=.ok () ↔ InputContract x w bias eps := by
  have unpack : InputContract x w bias eps ↔ TensorContract x Bool.true ∧ TensorContract w Bool.true ∧
      (∀ b,bias=some b → TensorContract b Bool.true) ∧ 0<x.shape.length ∧ 0<width x ∧ width x≤65536 ∧
      w.device=x.device ∧ (∀ b,bias=some b → b.device=x.device) ∧ width x≤w.numel ∧
      (∀ b,bias=some b → width x≤b.numel) ∧ eps.numeric=Bool.true ∧ eps.finite=Bool.true ∧ 0<eps.value :=
    ⟨fun h => ⟨h.input,h.weight,h.biasTensor,h.rank,h.positive,h.bounded,h.weightDevice,h.biasDevice,
      h.weightCount,h.biasCount,h.epsilon⟩,
      fun ⟨a,b,c,d,e,f,g,h,i,j,k,l,m⟩ => ⟨a,b,c,d,e,f,g,h,i,j,⟨k,l,m⟩⟩⟩
  rw [unpack]
  cases bias with
  | none =>
    cases hx : checkTensor x Bool.true <;> cases hw : checkTensor w Bool.true <;>
      simp [checkInputs,checkOptional,←checkTensor_iff,hx,hw,ite_eq_iff,Nat.pos_iff_ne_zero,and_assoc]
  | some b =>
    cases hx : checkTensor x Bool.true <;> cases hw : checkTensor w Bool.true <;>
      cases hb : checkTensor b Bool.true <;>
      simp [checkInputs,checkOptional,←checkTensor_iff,hx,hw,hb,ite_eq_iff,Nat.pos_iff_ne_zero,and_assoc,and_left_comm,and_comm]

theorem checkGradient_iff (dy x : TensorMetadata) :
    checkGradient dy x=.ok () ↔ GradientContract dy x := by
  have unpack : GradientContract dy x ↔ TensorContract dy Bool.false ∧ dy.device=x.device ∧ dy.shape=x.shape :=
    ⟨fun h => ⟨h.tensor,h.device,h.shape⟩,fun ⟨a,b,c⟩ => ⟨a,b,c⟩⟩
  rw [unpack]
  simp only [←checkTensor_iff]
  cases hd : checkTensor dy Bool.false <;> simp [checkGradient,hd,ite_eq_iff,and_assoc]

theorem checkArguments_iff (x w : ArgumentMetadata) (bias : Option ArgumentMetadata) (eps : Epsilon) :
    checkArguments x w bias eps=.ok () ↔ AdmittedArguments x w bias eps := by
  cases x <;> cases w <;> cases bias with
  | none => simp [checkArguments,AdmittedArguments,checkInputs_iff]
  | some b => cases b <;> simp [checkArguments,AdmittedArguments,checkInputs_iff]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.Contract
