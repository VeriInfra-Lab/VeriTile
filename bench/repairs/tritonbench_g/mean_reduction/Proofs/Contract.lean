import bench.repairs.tritonbench_g.mean_reduction.Contract
import bench.tritonbench_g.mean_reduction.Proofs.SourceIndex

namespace VeriTile.Bench.Repairs.MeanReduction
open VeriTile.Bench.TritonBenchG.MeanReduction.Whole

theorem contract_iff (metadata : TensorMetadata) (shape : Fin rank → Nat)
    (dims : List (Fin rank)) (dtype : Option DType) : InputContract metadata shape dims dtype ↔
    metadata.layout = .strided ∧ metadata.dtype = .floating .fp32 ∧
    (dtype = none ∨ dtype = some (.floating .fp32)) ∧ metadata.device.kind = "cuda" ∧
    metadata.isNeg = Bool.false ∧ (0 < originalNumel shape → metadata.dataPtr.toNat % 4 = 0) ∧
    dims.Nodup ∧ 0 < columns shape dims ∧ ((rows shape dims+7)/8)*8 < 2^31 ∧
    ((columns shape dims+7)/8)*8 < 2^31 ∧
    (0 < rows shape dims → (((rows shape dims+7)/8)*8-1)*columns shape dims < 2^31) := by
  constructor
  · intro h; exact ⟨h.layout,h.inputDType,h.outputDType,h.cuda,h.noNeg,h.alignment,
      h.distinct,h.positive,h.paddedRows,h.paddedCols,h.rowProduct⟩
  · rintro ⟨hl,hi,ho,hc,hn,ha,hd,hp,hm,hN,hr⟩; exact ⟨hl,hi,ho,hc,hn,ha,hd,hp,hm,hN,hr⟩

theorem checkNormalized_iff (metadata : TensorMetadata) (shape : Fin rank → Nat)
    (dims : List (Fin rank)) (dtype : Option DType) :
    checkNormalized metadata shape dims dtype = .ok () ↔ InputContract metadata shape dims dtype := by
  rw [contract_iff]
  simp [checkNormalized,VeriTile.Torch.TensorMetadata.isCuda,ite_eq_iff,Nat.pos_iff_ne_zero]
  tauto

theorem checkInput_iff (metadata : TensorMetadata) (shape : Fin rank → Nat)
    (axes : Axes) (dtype : Option DType) (dims : List (Fin rank)) :
    checkInput metadata shape axes dtype = .ok dims ↔ Admitted metadata shape axes dtype dims := by
  unfold checkInput Admitted
  cases hn : normalize rank axes with
  | none => simp
  | some parsed =>
    cases hc : checkNormalized metadata shape parsed dtype with
    | error error =>
      simp only [bind,pure,Except.bind,Except.pure,hc,reduceCtorEq,Option.some.injEq,false_iff]
      rintro ⟨eq,h⟩
      cases eq
      rw [(checkNormalized_iff metadata shape _ dtype).mpr h] at hc
      cases hc
    | ok value =>
      cases value
      have h := (checkNormalized_iff metadata shape parsed dtype).mp hc
      simp only [bind,pure,Except.bind,Except.pure,hc,Except.ok.injEq,Option.some.injEq]
      constructor
      · intro eq; subst dims; exact ⟨rfl,h⟩
      · exact And.left

/-- The source's signed arithmetic is derived from exactly the checked extents.
No numerical operand condition occurs here. -/
theorem index_bounds (h : InputContract metadata shape dims dtype)
    (pid t lane col : Nat) (hp : pid < gridSize (rows shape dims))
    (ht : t < gridSize (columns shape dims)) (hl : lane < 8) (hc : col < 8) :
    rows shape dims < 2^31 ∧ columns shape dims < 2^31 ∧
    pid*8+lane < 2^31 ∧ t*8+col < 2^31 ∧ t*8+8 < 2^31 ∧
    (pid*8+lane)*columns shape dims < 2^31 := by
  have hr := h.paddedRows
  have hn := h.paddedCols
  have nonempty : 0 < rows shape dims := by unfold gridSize at hp;omega
  have hp' : pid*8+lane ≤ ((rows shape dims+7)/8)*8-1 := by unfold gridSize at hp;omega
  have hm := Nat.mul_le_mul_right (columns shape dims) hp'
  have hi := h.rowProduct nonempty
  unfold gridSize at hp ht
  omega

theorem checkArgument_iff (argument : VeriTile.Torch.ArgumentMetadata)
    (shape : Fin rank → Nat) (axes : Axes) (dtype : Option DType) (dims : List (Fin rank)) :
    checkArgument argument shape axes dtype=.ok dims ↔
      ∃ metadata, argument=.tensor metadata ∧ Admitted metadata shape axes dtype dims := by
  cases argument with
  | nonTensor => simp [checkArgument]
  | tensor metadata => simp [checkArgument,checkInput_iff]

theorem native_masks (h : InputContract metadata shape dims dtype)
    (pid t lane col : Nat) (hp : pid < gridSize (rows shape dims))
    (ht : t < gridSize (columns shape dims)) (hl : lane < 8) (hc : col < 8) :
    ((rowBits pid lane).slt (i32 (rows shape dims))=Bool.true ↔ pid*8+lane<rows shape dims) ∧
    ((colBits (t*8) col).slt (i32 (columns shape dims))=Bool.true ↔ t*8+col<columns shape dims) := by
  have bounds := index_bounds h pid t lane col hp ht hl hc
  simp only [BitVec.slt_iff_toInt_lt,rowBits_exact pid lane bounds.2.2.1,
    colBits_exact (t*8) col bounds.2.2.2.1,i32_exact bounds.1,i32_exact bounds.2.1]
  omega

/-- Original strided coordinates and the observed data pointer refer to the
 same bytes, including nonzero storage offsets. Empty inputs impose no unused
 pointer/alignment condition; valid backing addresses are ordinary Tensor laws. -/
theorem original_pointer (binding : TensorBinding metadata x)
    (nonempty : 0 < originalNumel x.shape) (coordinate : Coordinates x.shape) :
    (metadata.dataPtr + BitVec.ofNat 64
      (4*(∑ ax, x.stride ax * (coordinate ax).val))).toNat =
      4*x.address coordinate := by
  rw [binding.pointer nonempty,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
  have bound := binding.addressFits coordinate
  have sum : 4*x.base+4*(∑ ax, x.stride ax*(coordinate ax).val) =
      4*x.address coordinate := by simp [Tensor.address,Nat.mul_add]
  rw [sum]
  exact Nat.mod_eq_of_lt (by omega)

#axiomsClean original_pointer
/-- The checked signed arithmetic feeds exact active byte addresses. The
 backing spans are ordinary storage facts, not padded-allocation requirements.
 Load/store modeling references are in documents/GpuMemoryModel.md. -/
theorem checked_pointers (h : InputContract metadata shape dims dtype)
    (inputBase outputBase : Nat)
    (inputEnd : 4*(inputBase+rows shape dims*columns shape dims) ≤ 2^64)
    (outputEnd : 4*(outputBase+rows shape dims) ≤ 2^64)
    (pid t lane col : Nat) (hp : pid < gridSize (rows shape dims))
    (ht : t < gridSize (columns shape dims)) (hl : lane < 8) (hc : col < 8)
    (row : pid*8+lane < rows shape dims) (column : t*8+col < columns shape dims) :
    (nativeInputPointer inputBase pid lane (columns shape dims) (t*8) col).toNat =
      4*(inputBase+(pid*8+lane)*columns shape dims+(t*8+col)) ∧
    (nativeOutputPointer outputBase pid lane).toNat = 4*(outputBase+pid*8+lane) := by
  have ints := index_bounds h pid t lane col hp ht hl hc
  have flat : (pid*8+lane)*columns shape dims+(t*8+col) <
      rows shape dims*columns shape dims := by nlinarith
  have ib : 4*(inputBase+(pid*8+lane)*columns shape dims+(t*8+col)) < 2^64 := by omega
  have ob : 4*(outputBase+pid*8+lane) < 2^64 := by omega
  simp only [nativeInputPointer,nativeOutputPointer,rowBits_exact pid lane ints.2.2.1,
    colBits_exact (t*8) col ints.2.2.2.1,
    rowProductBits_exact pid lane _ ints.2.2.2.2.2,Int.toNat_natCast,
    BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
  constructor
  · convert Nat.mod_eq_of_lt ib using 1 <;> congr 1 <;> omega
  · convert Nat.mod_eq_of_lt ob using 1 <;> congr 1 <;> omega

#axiomsClean checked_pointers
#axiomsClean checkNormalized_iff
#axiomsClean checkInput_iff
#axiomsClean index_bounds
#axiomsClean checkArgument_iff
#axiomsClean native_masks
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.MeanReduction
