import Mathlib
import bench.repairs.tritonbench_g.dequantize_rowwise.Contract
import bench.repairs.tritonbench_g.dequantize_rowwise.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.DequantizeRowwise

theorem contract_iff (x s : TensorMetadata) : InputContract x s ↔
    (x.layout=.strided ∧ s.layout=.strided) ∧
    (x.dtype=.other "int8" ∧ s.dtype=.floating .fp32) ∧
    (x.device.kind="cuda" ∧ s.device.kind="cuda") ∧ x.device=s.device ∧
    (x.shape.length=2 ∧ s.shape.length=1) ∧
    (x.isContiguous=Bool.true ∧ s.isContiguous=Bool.true) ∧
    (x.isNeg=Bool.false ∧ s.isNeg=Bool.false) ∧ columns x≤2^20 ∧ rows x<2^31 ∧
    (0<x.numel → rows x≤ s.numel) ∧ (0<x.numel → s.dataPtr.toNat%4=0) := by
  constructor
  · intro h;exact ⟨h.layout,h.dtype,h.cuda,h.sameDevice,h.rank,h.contiguous,h.noNeg,h.width,h.grid,h.scales,h.alignment⟩
  · rintro ⟨a,b,c,d,e,f,g,h,i,j,k⟩;exact ⟨a,b,c,d,e,f,g,h,i,j,k⟩

theorem checkTensorInputs_iff (x s : TensorMetadata) : checkTensorInputs x s=.ok () ↔ InputContract x s := by
  rw [contract_iff]
  simp [checkTensorInputs,VeriTile.Torch.TensorMetadata.isCuda,ite_eq_iff,and_assoc]

theorem checkInputs_iff (x s : ArgumentMetadata) : checkInputs x s=.ok () ↔ AdmittedArguments x s := by
  cases x <;> cases s <;> simp [checkInputs,AdmittedArguments,checkTensorInputs_iff]

theorem shape_eq (x s : TensorMetadata) (h : InputContract x s) : x.shape=[rows x,columns x] := by
  rcases x with ⟨dtype,layout,device,shape,contiguous,neg,ptr⟩
  rcases shape with _|⟨a,shape⟩
  · have := h.rank.1;simp at this
  rcases shape with _|⟨b,tail⟩
  · have := h.rank.1;simp at this
  cases tail with
  | nil => rfl
  | cons c tail => have := h.rank.1;simp at this

theorem numel_eq (x s : TensorMetadata) (h : InputContract x s) : x.numel=rows x*columns x := by
  unfold VeriTile.Torch.TensorMetadata.numel
  rw [shape_eq x s h]
  simp

def paddedWidth (x : TensorMetadata) : Nat := 2^Nat.clog 2 (columns x)
theorem padded_covers (x : TensorMetadata) : columns x≤paddedWidth x := Nat.le_pow_clog (by decide) _
theorem padded_bound (x s : TensorMetadata) (h : InputContract x s) : paddedWidth x≤2^20 :=
  Nat.pow_le_pow_right (by decide) (Nat.clog_le_of_le_pow h.width)
def parameters (x : TensorMetadata) : Parameters := ⟨rows x,columns x,paddedWidth x,padded_covers x⟩

theorem numel_bound (x s : TensorMetadata) (h : InputContract x s) : x.numel<2^51 := by
  rw [numel_eq x s h]
  have grid := h.grid
  have := Nat.mul_le_mul (by omega : rows x≤2^31-1) h.width
  omega

#axiomsClean checkInputs_iff
#axiomsClean numel_bound
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DequantizeRowwise
