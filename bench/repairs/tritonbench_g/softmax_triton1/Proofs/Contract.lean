import bench.repairs.tritonbench_g.softmax_triton1.Contract
import bench.repairs.tritonbench_g.square_matrix.Proofs.Geometry

namespace VeriTile.Bench.Repairs.SoftmaxTriton1

theorem blockSize_bounds (n : Nat) : 0<blockSize n ∧ n≤blockSize n :=
  ⟨Nat.lt_of_lt_of_le (by decide : 0<1) (Nat.le_max_left _ _),
   (VeriTile.Bench.Repairs.SquareMatrix.block_covers n).trans (Nat.le_max_right _ _)⟩

namespace Contiguous

theorem contract_iff (x : TensorMetadata) : InputContract x ↔
    x.layout = .strided ∧ x.dtype = .floating .fp32 ∧ x.device.kind = "cuda" ∧
    x.shape.length = 2 ∧ x.isContiguous = Bool.true ∧ x.isNeg = Bool.false ∧
    (0 < x.numel → rows x < 2^31) ∧ (0 < x.numel → x.dataPtr.toNat % 4 = 0) ∧
    (0 < x.numel → block (cols x) ≤ 2^20) := by
  constructor
  · intro h; exact ⟨h.layout,h.dtype,h.cuda,h.rank,h.contiguous,h.noNeg,h.grid,h.alignment,h.frontend⟩
  · rintro ⟨a,b,c,d,e,f,g,h,i⟩; exact ⟨a,b,c,d,e,f,g,h,i⟩

theorem checkTensorInput_iff (x : TensorMetadata) :
    checkTensorInput x = .ok () ↔ InputContract x := by
  rw [contract_iff]
  simp [checkTensorInput, VeriTile.Torch.TensorMetadata.isCuda, ite_eq_iff]

theorem checkInput_iff (x : ArgumentMetadata) :
    checkInput x = .ok () ↔ AdmittedArgument x := by
  cases x <;> simp [checkInput,AdmittedArgument,checkTensorInput_iff]

theorem shape_eq (x : TensorMetadata) (h : InputContract x) : x.shape=[rows x,cols x] := by
  obtain ⟨r,c,hs⟩ := List.length_eq_two.mp h.rank
  simp [rows,cols,hs]

theorem numel_eq (x : TensorMetadata) (h : InputContract x) : x.numel=rows x*cols x := by
  simp [VeriTile.Torch.TensorMetadata.numel,shape_eq x h]

theorem block_eq (x : TensorMetadata) (h : InputContract x) (positive : 0<x.numel) :
    blockSize (cols x)=block (cols x) := by
  have hc : 0<cols x := by have := numel_eq x h; nlinarith
  have cover := VeriTile.Bench.Repairs.SquareMatrix.block_covers (cols x)
  exact Nat.max_eq_right (by omega)

theorem launch_bounds (x : TensorMetadata) (h : InputContract x) (positive : 0<x.numel)
    (r c : Nat) (hr : r<rows x) (hc : c<blockSize (cols x)) :
    r<2^31 ∧ c<2^31 ∧ cols x<2^63 := by
  have hb := h.frontend positive
  rw [←block_eq x h positive] at hb
  have cover := (blockSize_bounds (cols x)).2
  exact ⟨hr.trans (h.grid positive), by omega, by omega⟩

#axiomsClean checkTensorInput_iff
#axiomsClean checkInput_iff
#axiomsClean launch_bounds
end Contiguous
end VeriTile.Bench.Repairs.SoftmaxTriton1
