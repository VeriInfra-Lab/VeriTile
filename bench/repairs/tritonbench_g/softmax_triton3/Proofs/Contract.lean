import bench.repairs.tritonbench_g.softmax_triton3.Contract
import bench.repairs.tritonbench_g.square_matrix.Proofs.Geometry
namespace VeriTile.Bench.Repairs.SoftmaxTriton3

theorem block_bounds (n : Nat) : 0<blockSize n ∧ n≤blockSize n :=
  ⟨lt_of_lt_of_le (by decide : 0<2) (Nat.le_max_right _ _),
   (VeriTile.Bench.Repairs.SquareMatrix.block_covers n).trans (Nat.le_max_left _ _)⟩

namespace Contiguous

theorem tensor_iff (x : TensorMetadata) : TensorContract x ↔
    x.layout = .strided ∧ x.dtype = .floating .fp32 ∧ x.device.kind = "cuda" ∧
    x.shape ≠ [] ∧ x.isContiguous = Bool.true ∧ x.isNeg = Bool.false ∧
    (0 < x.numel → rows x < 2^31) ∧ (0 < x.numel → x.dataPtr.toNat % 4 = 0) ∧
    (0 < x.numel → blockSize (cols x) ≤ 2^20) := by
  constructor
  · intro h;exact ⟨h.layout,h.dtype,h.cuda,h.rank,h.contiguous,h.noNeg,h.grid,h.alignment,h.frontend⟩
  · rintro ⟨a,b,c,d,e,f,g,h,i⟩;exact ⟨a,b,c,d,e,f,g,h,i⟩

theorem checkTensor_iff (x : TensorMetadata) : checkTensor x=.ok () ↔ TensorContract x := by
  rw [tensor_iff]
  simp [checkTensor,VeriTile.Torch.TensorMetadata.isCuda,ite_eq_iff]

theorem checkTyped_iff (x : TensorMetadata) (mask : Option TensorMetadata) (dim : Int) :
    checkTyped x mask dim=.ok () ↔ InputContract x mask dim := by
  have unpack : InputContract x mask dim ↔ TensorContract x ∧
      (dim = -1 ∨ dim = (x.shape.length : Int)-1) ∧
      ∀ m,mask=some m → TensorContract m ∧ m.device=x.device ∧ cols m=cols x ∧ rows m=rows x := by
    constructor
    · intro h;exact ⟨h.input,h.dimension,h.mask⟩
    · rintro ⟨a,b,c⟩;exact ⟨a,b,c⟩
  rw [unpack]
  simp only [←checkTensor_iff]
  cases hx : checkTensor x with
  | error e => simp [checkTyped,hx]
  | ok unit =>
    cases unit
    cases mask with
    | none => simp [checkTyped,hx,ite_eq_iff,not_and_or] <;> tauto
    | some m =>
      cases hm : checkTensor m with
      | error e => simp [checkTyped,hx,hm,ite_eq_iff,not_and_or] <;> tauto
      | ok unit => cases unit; simp [checkTyped,hx,hm,ite_eq_iff,not_and_or] <;> tauto

theorem checkInput_iff (x : ArgumentMetadata) (mask : Option ArgumentMetadata) (dim : Int) :
    checkInput x mask dim=.ok () ↔ AdmittedArguments x mask dim := by
  cases x <;> cases mask with
  | none => simp [checkInput,AdmittedArguments,checkTyped_iff]
  | some m => cases m <;> simp [checkInput,AdmittedArguments,checkTyped_iff]

theorem numel_eq (x : TensorMetadata) (h : x.shape≠[]) : x.numel=rows x*cols x := by
  have hs := List.dropLast_concat_getLast h
  rw [VeriTile.Torch.TensorMetadata.numel,←hs,List.prod_append]
  simp [rows,cols,List.getLast?_eq_some_getLast h]

#axiomsClean checkTensor_iff
#axiomsClean checkTyped_iff
#axiomsClean checkInput_iff
#axiomsClean numel_eq
end Contiguous
end VeriTile.Bench.Repairs.SoftmaxTriton3
