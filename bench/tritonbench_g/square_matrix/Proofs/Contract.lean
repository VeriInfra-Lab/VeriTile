import bench.tritonbench_g.square_matrix.Proofs.Geometry

namespace VeriTile.Bench.Repairs.SquareMatrix

theorem contract_iff (x : TensorMetadata) : InputContract x ↔
    x.layout = .strided ∧ x.dtype = .floating .fp32 ∧ x.device.kind = "cuda" ∧
    x.shape.length = 2 ∧ x.isContiguous = Bool.true ∧ x.isNeg = Bool.false ∧ x.numel < 2^31 ∧
    (0 < x.numel → x.dataPtr.toNat % 4 = 0) ∧
    (0 < x.numel → block (cols x) ≤ 2^20) := by
  constructor
  · intro h; exact ⟨h.layout,h.dtype,h.cuda,h.rank,h.contiguous,h.noNeg,h.extent,h.alignment,h.frontend⟩
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

#axiomsClean checkTensorInput_iff
#axiomsClean checkInput_iff
#axiomsClean numel_eq
end VeriTile.Bench.Repairs.SquareMatrix
