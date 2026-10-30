import bench.tritonbench_g.layer_norm_liger.Proofs.HostContract
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic

namespace VeriTile.Bench.Repairs.LayerNormLiger.Contract
open VeriTile.Torch
noncomputable section

theorem checkTensor_iff (m : TensorMetadata) : checkTensor m = .ok () ↔ TensorContract m := by
  have unpack : TensorContract m ↔ m.layout = .strided ∧ m.dtype = .floating .fp32 ∧
      m.device.kind = "cuda" ∧ m.isNeg = Bool.false ∧ (0 < m.numel → m.dataPtr.toNat % 4 = 0) :=
    ⟨fun h => ⟨h.layout,h.dtype,h.cuda,h.noNeg,h.alignment⟩,fun ⟨a,b,c,d,e⟩ => ⟨a,b,c,d,e⟩⟩
  rw [unpack]
  simp [checkTensor, ite_eq_iff, and_assoc]

theorem checkInputs_iff (x w b : TensorMetadata) (eps : Epsilon) :
    checkInputs x w b eps = .ok () ↔ InputContract x w b eps := by
  have unpack : InputContract x w b eps ↔ TensorContract x ∧ TensorContract w ∧ TensorContract b ∧
      0 < x.shape.length ∧ 0 < x.numel ∧ 0 < width x ∧ width x ≤ 65536 ∧
      w.shape = [width x] ∧ b.shape = [width x] ∧ w.device = x.device ∧ b.device = x.device ∧
      eps.numeric = Bool.true ∧ eps.finite = Bool.true ∧ 0 < eps.value :=
    ⟨fun h => ⟨h.input,h.weight,h.bias,h.rank,h.nonempty,h.width_pos,h.width_bound,h.weightShape,h.biasShape,
      h.device.1,h.device.2,h.epsilon⟩,fun ⟨a,b,c,d,e,f,g,h,i,j,k,l,m,n⟩ => ⟨a,b,c,d,e,f,g,h,i,⟨j,k⟩,⟨l,m,n⟩⟩⟩
  rw [unpack]
  simp only [← checkTensor_iff]
  cases hx : checkTensor x <;> cases hw : checkTensor w <;> cases hb : checkTensor b <;>
    simp [checkInputs,hx,hw,hb,ite_eq_iff,Nat.pos_iff_ne_zero,and_assoc]

theorem checkGradient_iff (dy x : TensorMetadata) :
    checkGradient dy x = .ok () ↔ GradientContract dy x := by
  have unpack : GradientContract dy x ↔ TensorContract dy ∧ 0 < dy.shape.length ∧
      dy.device = x.device ∧ dy.numel = x.numel ∧ width dy = width x :=
    ⟨fun h => ⟨h.tensor,h.rank,h.device,h.elements,h.columns⟩,fun ⟨a,b,c,d,e⟩ => ⟨a,b,c,d,e⟩⟩
  rw [unpack]
  simp only [← checkTensor_iff]
  cases hd : checkTensor dy <;> simp [checkGradient,hd,ite_eq_iff,Nat.pos_iff_ne_zero,and_assoc]

theorem checkArguments_iff (x w b : ArgumentMetadata) (eps : Epsilon) :
    checkArguments x w b eps = .ok () ↔ AdmittedArguments x w b eps := by
  cases x <;> cases w <;> cases b <;> simp [checkArguments,AdmittedArguments,checkInputs_iff]

theorem shape_split (x : TensorMetadata) (h : 0 < x.shape.length) :
    x.shape = x.shape.dropLast ++ [width x] := by
  unfold width
  have hn : x.shape ≠ [] := by intro hz; simp [hz] at h
  have split := List.dropLast_concat_getLast hn
  have last : x.shape.getLast hn = x.shape.getLastD 0 := by
    have eq := List.getLastD_concat (a := 0) (b := x.shape.getLast hn) (l := x.shape.dropLast)
    rw [split] at eq
    exact eq.symm
  rw [last] at split
  exact split.symm

theorem shape_elements (x : TensorMetadata) (h : 0 < x.shape.length) :
    x.numel = rows x * width x := by
  conv_lhs => unfold TensorMetadata.numel; rw [shape_split x h]
  simp [rows]

theorem rows_positive (x w b : TensorMetadata) (eps : Epsilon) (h : InputContract x w b eps) :
    0 < rows x := by
  have hn := h.nonempty
  rw [shape_elements x h.rank] at hn
  by_contra hz
  have zero : rows x = 0 := by omega
  simp [zero] at hn

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormLiger.Contract
