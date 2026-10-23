import bench.tritonbench_g.mean_reduction.Proofs.SourceIndex
import bench.tritonbench_g.mean_reduction.Proofs.WholeHost

namespace VeriTile.Bench.TritonBenchG.MeanReduction.Whole
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000

def paddedRows (M : Nat) : Nat := gridSize M*8
def paddedCols (N : Nat) : Nat := gridSize N*8

def maxInputOffset (M N : Nat) : Nat := (paddedRows M-1)*N+(paddedCols N-1)

/-- Bounds admit every active logical tensor layout. They constrain actual
signed32 arithmetic and full64-bit aligned addresses, including inactive lanes
whose pointer expressions are still formed. No numerical value restriction. -/
structure FiniteAdmission {α : Type} {rank : Nat} {x : Tensor rank}
    {dims : List (Fin rank)} {nodup : dims.Nodup} {entry : Nat → α} {preexisting : Nat → Prop}
    {c : Contiguous α x dims nodup entry preexisting} (a : Allocated c) where
  reductionPositive : 0 < columns x.shape dims
  rowsI32 : rows x.shape dims < 2^31
  columnsI32 : columns x.shape dims < 2^31
  paddedRowsI32 : paddedRows (rows x.shape dims) < 2^31
  paddedColsI32 : paddedCols (columns x.shape dims) < 2^31
  rowProductI32 : rows x.shape dims = 0 ∨
    (paddedRows (rows x.shape dims)-1)*columns x.shape dims < 2^31
  originalStart : Nat
  originalEnd : Nat
  originalBounds : ∀ i, originalStart ≤ x.address i ∧ x.address i < originalEnd
  originalProtected : ∀ addr, originalStart ≤ addr → addr < originalEnd → preexisting addr
  originalByteEnd : 4*originalEnd ≤ 2^64
  compressedByteEnd : 4*(c.base+rows x.shape dims*columns x.shape dims) ≤ 2^64
  outputByteEnd : 4*(a.base+rows x.shape dims) ≤ 2^64
  inputPaddedEnd : rows x.shape dims = 0 ∨
    4*(c.base+maxInputOffset (rows x.shape dims) (columns x.shape dims))+4 ≤ 2^64
  outputPaddedEnd : rows x.shape dims = 0 ∨
    4*(a.base+(paddedRows (rows x.shape dims)-1))+4 ≤ 2^64

variable {α : Type} {rank : Nat} {x : Tensor rank} {dims : List (Fin rank)}
  {nodup : dims.Nodup} {entry : Nat → α} {preexisting : Nat → Prop}
  {c : Contiguous α x dims nodup entry preexisting}

theorem padded_row_bound (M pid lane : Nat) (hp : pid < gridSize M) (hl : lane < 8) :
    pid*8+lane < paddedRows M := by unfold paddedRows; omega

theorem padded_col_bound (N t col : Nat) (ht : t < gridSize N) (hc : col < 8) :
    t*8+col < paddedCols N := by unfold paddedCols; omega

/-- Every recorded source arithmetic intermediate has its exact signed32
interpretation, and the recorded predicates equal the mathematical masks. -/
theorem source_integer_correspondence (a : Allocated c) (f : FiniteAdmission a)
    (pid t lane col : Nat) (hp : pid < gridSize (rows x.shape dims))
    (ht : t < gridSize (columns x.shape dims)) (hl : lane < 8) (hc : col < 8) :
    (rowBits pid lane).toInt = ((pid*8+lane : Nat) : Int) ∧
    (colBits (t*8) col).toInt = ((t*8+col : Nat) : Int) ∧
    (rowProductBits pid lane (columns x.shape dims)).toInt =
      (((pid*8+lane)*columns x.shape dims : Nat) : Int) ∧
    ((rowBits pid lane).slt (i32 (rows x.shape dims)) = Bool.true ↔ pid*8+lane < rows x.shape dims) ∧
    ((colBits (t*8) col).slt (i32 (columns x.shape dims)) = Bool.true ↔ t*8+col < columns x.shape dims) ∧
    (∀ v ∈ [rows x.shape dims,columns x.shape dims,8,pid,pid*8,pid*8+lane,
      (pid*8+lane)*columns x.shape dims,t,t*8,col,t*8+col,t*8+8], (i32 v).toInt = (v : Int)) := by
  have hr := padded_row_bound (rows x.shape dims) pid lane hp hl
  have hs := padded_col_bound (columns x.shape dims) t col ht hc
  have nonempty : rows x.shape dims ≠ 0 := by intro h; simp [h,gridSize] at hp
  have hi := f.rowProductI32.resolve_left nonempty
  have upper : (pid*8+lane)*columns x.shape dims ≤
      (paddedRows (rows x.shape dims)-1)*columns x.shape dims :=
    Nat.mul_le_mul_right (columns x.shape dims) (by omega)
  have hr32 : pid*8+lane < 2^31 := lt_trans hr f.paddedRowsI32
  have hs32 : t*8+col < 2^31 := lt_trans hs f.paddedColsI32
  refine ⟨rowBits_exact pid lane hr32,colBits_exact (t*8) col hs32,
    rowProductBits_exact pid lane _ (lt_of_le_of_lt upper hi),?_,?_,?_⟩
  · rw [BitVec.slt_iff_toInt_lt,rowBits_exact pid lane hr32,i32_exact f.rowsI32]
    omega
  · rw [BitVec.slt_iff_toInt_lt,colBits_exact (t*8) col hs32,i32_exact f.columnsI32]
    omega
  · intro v hv
    apply i32_exact
    have loopEnd : t*8+8 ≤ paddedCols (columns x.shape dims) := by unfold paddedCols; omega
    have hp32 := f.paddedRowsI32
    have hc32 := f.paddedColsI32
    have hm32 := f.rowsI32
    have hn32 := f.columnsI32
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hv
    rcases hv with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> omega

/-- Native pointers include all padded input and output lanes; active accesses
also include the last byte of each four-byte cell and the allocation end. -/
theorem source_pointer_correspondence (a : Allocated c) (f : FiniteAdmission a)
    (pid t lane col : Nat) (hp : pid < gridSize (rows x.shape dims))
    (ht : t < gridSize (columns x.shape dims)) (hl : lane < 8) (hc : col < 8) :
    (bytePointer c.base ((pid*8+lane)*columns x.shape dims+(t*8+col))).toNat =
      4*(c.base+(pid*8+lane)*columns x.shape dims+(t*8+col)) ∧
    (bytePointer a.base (pid*8+lane)).toNat = 4*(a.base+(pid*8+lane)) := by
  have hr := padded_row_bound (rows x.shape dims) pid lane hp hl
  have hs := padded_col_bound (columns x.shape dims) t col ht hc
  have nonempty : rows x.shape dims ≠ 0 := by intro h; simp [h,gridSize] at hp
  have inputEnd := f.inputPaddedEnd.resolve_left nonempty
  have outputEnd := f.outputPaddedEnd.resolve_left nonempty
  have upper : (pid*8+lane)*columns x.shape dims+(t*8+col) ≤
      maxInputOffset (rows x.shape dims) (columns x.shape dims) := by
    unfold maxInputOffset
    have hm := Nat.mul_le_mul_right (columns x.shape dims) (show pid*8+lane ≤ paddedRows (rows x.shape dims)-1 by omega)
    omega
  constructor
  · simpa only [Nat.add_assoc] using bytePointer_exact c.base
      ((pid*8+lane)*columns x.shape dims+(t*8+col)) (by omega)
  · exact bytePointer_exact a.base (pid*8+lane) (by omega)

/-- Complete active FP32 words and their final bytes are in the allocated
buffers; aligned words cannot overlap each other or preexisting words. -/
theorem active_accesses (a : Allocated c) (f : FiniteAdmission a)
    (row col : Nat) (hr : row < rows x.shape dims) (hc : col < columns x.shape dims) :
    row*columns x.shape dims+col < rows x.shape dims*columns x.shape dims ∧
    4*(c.base+row*columns x.shape dims+col)+4 ≤ 2^64 ∧
    4*(a.base+row)+4 ≤ 2^64 ∧
    4 ∣ 4*(c.base+row*columns x.shape dims+col) ∧ 4 ∣ 4*(a.base+row) := by
  have inputBound : row*columns x.shape dims+col < rows x.shape dims*columns x.shape dims := by nlinarith
  have ib := f.compressedByteEnd
  have ob := f.outputByteEnd
  exact ⟨inputBound,by omega,by omega,dvd_mul_right _ _,dvd_mul_right _ _⟩

theorem original_accesses (a : Allocated c) (f : FiniteAdmission a)
    (i : Coordinates x.shape) :
    4*(x.address i)+4 ≤ 2^64 ∧ preexisting (x.address i) := by
  have h := f.originalBounds i
  have hb := f.originalByteEnd
  exact ⟨by omega,f.originalProtected _ h.1 h.2⟩

theorem output_preexisting_bytes_separate (a : Allocated c) (row addr i j : Nat)
    (hr : row < rows x.shape dims) (hp : preexisting addr) (hi : i < 4) (hj : j < 4) :
    4*(a.base+row)+i ≠ 4*addr+j := by
  intro equal
  have words : a.base+row = addr := by omega
  exact a.fresh row hr (by simpa only [words] using hp)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MeanReduction.Whole
