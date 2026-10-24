import bench.repairs.tritonbench_g.rowwise_quantization_triton.Contract
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.Public
namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton

theorem contract_iff (x : TensorMetadata) : InputContract x ↔
    x.layout=.strided ∧ x.dtype=.floating .fp32 ∧ x.device.kind="cuda" ∧
    x.shape.length=2 ∧ x.isContiguous=Bool.true ∧ x.isNeg=Bool.false ∧
    (0<columns x ∧ columns x≤2^20) ∧ rows x<2^31 ∧ (0<x.numel → x.dataPtr.toNat%4=0) := by
  constructor
  · intro h;exact ⟨h.layout,h.dtype,h.cuda,h.rank,h.contiguous,h.noNeg,h.width,h.grid,h.alignment⟩
  · rintro ⟨a,b,c,d,e,f,g,h,i⟩;exact ⟨a,b,c,d,e,f,g,h,i⟩

theorem checkTensorInputs_iff (x : TensorMetadata) : checkTensorInputs x=.ok () ↔ InputContract x := by
  rw [contract_iff]
  simp [checkTensorInputs,VeriTile.Torch.TensorMetadata.isCuda,ite_eq_iff,Nat.pos_iff_ne_zero,and_assoc]

theorem checkInputs_iff (x : ArgumentMetadata) : checkInputs x=.ok () ↔ AdmittedArguments x := by
  cases x <;> simp [checkInputs,AdmittedArguments,checkTensorInputs_iff]

theorem shape_eq (x : TensorMetadata) (h : InputContract x) : x.shape=[rows x,columns x] := by
  rcases x with ⟨dtype,layout,device,shape,contiguous,neg,ptr⟩
  rcases shape with _|⟨a,shape⟩
  · have := h.rank; simp at this
  rcases shape with _|⟨b,tail⟩
  · have := h.rank; simp at this
  cases tail with
  | nil => rfl
  | cons c tail => have := h.rank; simp at this

theorem numel_eq (x : TensorMetadata) (h : InputContract x) : x.numel=rows x*columns x := by
  unfold VeriTile.Torch.TensorMetadata.numel
  rw [shape_eq x h]
  simp

def paddedWidth (x : TensorMetadata) : Nat := 2^Nat.clog 2 (columns x)
theorem padded_covers (x : TensorMetadata) : columns x≤paddedWidth x := Nat.le_pow_clog (by decide) _
theorem padded_bound (x : TensorMetadata) (h : InputContract x) : paddedWidth x≤2^20 := by
  exact Nat.pow_le_pow_right (by decide) (Nat.clog_le_of_le_pow h.width.2)

/-- All evaluated row products and padded tail indices fit signed64. This
includes inactive tail lanes, rather than assuming their address arithmetic. -/
theorem native_indices (x : TensorMetadata) (h : InputContract x) (r lane : Nat)
    (hr : r<rows x) (hl : lane<paddedWidth x) :
    r<2^31 ∧ lane<2^31 ∧ r*columns x+lane<2^52 := by
  have rw : r<2^31 := hr.trans h.grid
  have lw := padded_bound x h
  have prod : r*columns x≤(2^31-1)*2^20 := Nat.mul_le_mul (by omega) h.width.2
  exact ⟨rw,by omega,by omega⟩

#axiomsClean checkInputs_iff
#axiomsClean native_indices
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton
