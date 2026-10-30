import bench.tritonbench_g.fused_layernorm_triton.Contract
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.Contract

theorem checkTensor_iff (m : Metadata) : checkTensor m = .ok () ↔ TensorContract m := by
  have unpack : TensorContract m ↔ m.layout = .strided ∧ m.dtype = .floating .fp32 ∧
      m.device.kind = "cuda" ∧ m.device.index = 0 ∧ m.isContiguous = Bool.true ∧
      m.storageOffset = 0 ∧ m.isNeg = Bool.false ∧
      (0 < m.numel → m.dataPtr.toNat % 4 = 0) :=
    ⟨fun h => ⟨h.layout,h.dtype,h.cuda,h.device,h.contiguous,h.offset,h.noNeg,h.alignment⟩,
     fun ⟨a,b,c,d,e,f,g,h⟩ => ⟨a,b,c,d,e,f,g,h⟩⟩
  rw [unpack]
  simp [checkTensor, ite_eq_iff]
  tauto

theorem checkInputs_iff (w b x : Metadata) : checkInputs w b x = .ok () ↔ InputContract w b x := by
  have unpack : InputContract w b x ↔ TensorContract w ∧ TensorContract b ∧ TensorContract x ∧
      x.shape.length = 2 ∧ 0 < width x ∧ w.shape = [width x] ∧ b.shape = [width x] :=
    ⟨fun h => ⟨h.weight,h.bias,h.input,h.rank,h.positive,h.weightShape,h.biasShape⟩,
     fun ⟨a,b,c,d,e,f,g⟩ => ⟨a,b,c,d,e,f,g⟩⟩
  rw [unpack]
  simp only [← checkTensor_iff]
  cases hw : checkTensor w <;> cases hb : checkTensor b <;> cases hx : checkTensor x <;>
    simp [checkInputs, hw, hb, hx, ite_eq_iff, Nat.pos_iff_ne_zero, and_assoc]

theorem checkArguments_iff (w b x : Argument) :
    checkArguments w b x = .ok () ↔ AdmittedArguments w b x := by
  cases w <;> cases b <;> cases x <;> simp [checkArguments, AdmittedArguments, checkInputs_iff]

theorem input_shape (x : Metadata) (h : x.shape.length = 2) : x.shape = [rows x, width x] := by
  rcases x with ⟨⟨dtype,layout,device,shape,contiguous,neg,ptr⟩,offset⟩
  cases shape with
  | nil => simp at h
  | cons s rest => cases rest with
    | nil => simp at h
    | cons d tail =>
      have e : tail = [] := by simpa using h
      subst tail
      rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.FusedLayernormTriton.Contract
