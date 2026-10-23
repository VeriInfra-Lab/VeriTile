import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.Contract

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.KernelIndex
open VeriTile.Torch

abbrev word (n : Nat) : BitVec 64 := BitVec.ofNat 64 n

def rowWord (c : KernelConfig) (pid lane : Nat) : BitVec 64 :=
  word pid * word c.blockRows + word lane

def columnWord (c : KernelConfig) (step lane : Nat) : BitVec 64 :=
  word step * word c.blockColumns + word lane

def matrixPointer {N M : Nat} (mem : KernelMemory N M) (c : KernelConfig)
    (pid lane col : Nat) : Nat → BitVec 64
  | 0 => word (4 * mem.matrixBase) + word 4 * (rowWord c pid lane * word M + word col)
  | step + 1 => matrixPointer mem c pid lane col step + word 4 * word c.blockColumns

def vectorPointer {N M : Nat} (mem : KernelMemory N M) (c : KernelConfig)
    (col : Nat) : Nat → BitVec 64
  | 0 => word (4 * mem.vectorBase) + word 4 * word col
  | step + 1 => vectorPointer mem c col step + word 4 * word c.blockColumns

def outputPointer {N M : Nat} (mem : KernelMemory N M) (c : KernelConfig)
    (pid lane : Nat) : BitVec 64 :=
  word (4 * mem.outputBase) + word 4 * rowWord c pid lane

def chunks (M : Nat) (c : KernelConfig) : Nat := (M + c.blockColumns - 1) / c.blockColumns

theorem config_bounds (c : KernelConfig) :
    0 < c.blockRows ∧ c.blockRows ≤ 8 ∧ 0 < c.blockColumns ∧ c.blockColumns ≤ 128 := by
  have hr := c.rows_mem
  have hc := c.columns_mem
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr hc
  omega

/-- Exactly the natural source-loop iterations reach a body. This connects
the ceil-div count used in finite bounds to the source's `m < M` guard. -/
theorem iterations (M : Nat) (c : KernelConfig) (step : Nat) :
    step < chunks M c ↔ step * c.blockColumns < M := by
  have positive := (config_bounds c).2.2.1
  have limit := Nat.div_mul_le_self (M + c.blockColumns - 1) c.blockColumns
  have cover : M ≤ chunks M c * c.blockColumns := by
    have divmod := Nat.mod_add_div (M + c.blockColumns - 1) c.blockColumns
    have rem := Nat.mod_lt (M + c.blockColumns - 1) positive
    rw [Nat.mul_comm c.blockColumns] at divmod
    unfold chunks
    omega
  constructor
  · intro active
    have le := Nat.mul_le_mul_right c.blockColumns (Nat.succ_le_iff.mpr active)
    rw [Nat.succ_mul] at le
    unfold chunks at le
    omega
  · intro active
    by_contra stopped
    have le := Nat.mul_le_mul_right c.blockColumns (Nat.le_of_not_gt stopped)
    omega

theorem word_exact {n : Nat} (h : n < 2^64) : (word n).toNat = n := by
  rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt h]

/-- Ordinary nonnegative scalar ABI, widened before arithmetic. -/
def scalarABI (n : Nat) : BitVec 64 :=
  if n < 2^31 then (BitVec.ofNat 32 n).signExtend 64 else word n

theorem i32_to_u64 (n : Nat) (h : n < 2^31) :
    (BitVec.ofNat 32 n).signExtend 64 = word n := by
  have hn : (BitVec.ofNat 32 n).toNat = n := by
    rw [BitVec.toNat_ofNat]
    exact Nat.mod_eq_of_lt (lt_trans h (by norm_num))
  have hd : 2 * (BitVec.ofNat 32 n).toNat < 2^32 := by rw [hn]; omega
  change BitVec.ofInt 64 (BitVec.ofNat 32 n).toInt = word n
  rw [BitVec.toInt_eq_toNat_of_lt hd, hn]
  exact BitVec.ofInt_natCast 64 n

theorem scalar_exact (n : Nat) (extent : n < 2^63) : (scalarABI n).toNat = n := by
  unfold scalarABI
  split
  · rename_i small
    rw [i32_to_u64 n small]
    exact word_exact (lt_trans extent (by norm_num))
  · exact word_exact (lt_trans extent (by norm_num))

def nativeRow (c : KernelConfig) (pid lane : Nat) : BitVec 64 :=
  (BitVec.ofNat 32 pid).signExtend 64 * word c.blockRows +
    (BitVec.ofNat 32 lane).signExtend 64

def nativeColumn (c : KernelConfig) (step col : Nat) : BitVec 64 :=
  word step * word c.blockColumns + (BitVec.ofNat 32 col).signExtend 64

theorem native_row (c : KernelConfig) (pid lane : Nat) (hp : pid < 2^31)
    (hl : lane < c.blockRows) : nativeRow c pid lane = rowWord c pid lane := by
  have bounds := config_bounds c
  simp only [nativeRow, rowWord, i32_to_u64 pid hp, i32_to_u64 lane (by omega)]

theorem native_column (c : KernelConfig) (step col : Nat) (hc : col < c.blockColumns) :
    nativeColumn c step col = columnWord c step col := by
  have bounds := config_bounds c
  simp only [nativeColumn, columnWord, i32_to_u64 col (by omega)]

/-- Signed interpretation of addptr's 64-bit offset preserves its byte residue. -/
theorem signed_pointer_offset (offset : BitVec 64) :
    BitVec.ofInt 64 (4 * offset.toInt) = word 4 * offset := by
  rw [BitVec.ofInt_mul, BitVec.ofInt_toInt]
  rfl

theorem row_modular (c : KernelConfig) (pid lane : Nat) :
    rowWord c pid lane = word (pid * c.blockRows + lane) := by
  simp only [rowWord, word, BitVec.ofNat_mul_ofNat, BitVec.ofNat_add_ofNat]

theorem column_modular (c : KernelConfig) (step lane : Nat) :
    columnWord c step lane = word (step * c.blockColumns + lane) := by
  simp only [columnWord, word, BitVec.ofNat_mul_ofNat, BitVec.ofNat_add_ofNat]

/-- Every formed pointer, including inactive lanes and final backedges, obeys
modular uint64 arithmetic. No bound on an inactive pointer is imposed. -/
theorem matrix_modular {N M : Nat} (mem : KernelMemory N M) (c : KernelConfig)
    (pid lane col step : Nat) :
    matrixPointer mem c pid lane col step =
      word (4 * (mem.matrixBase + (pid * c.blockRows + lane) * M + (step * c.blockColumns + col))) := by
  induction step with
  | zero =>
    simp only [matrixPointer, row_modular, word, BitVec.ofNat_mul_ofNat, BitVec.ofNat_add_ofNat]
    congr 1
    ring
  | succ step ih =>
    rw [matrixPointer, ih]
    simp only [word, BitVec.ofNat_mul_ofNat, BitVec.ofNat_add_ofNat]
    congr 1
    ring

theorem vector_modular {N M : Nat} (mem : KernelMemory N M) (c : KernelConfig)
    (col step : Nat) :
    vectorPointer mem c col step = word (4 * (mem.vectorBase + (step * c.blockColumns + col))) := by
  induction step with
  | zero =>
    simp only [vectorPointer, word, BitVec.ofNat_mul_ofNat, BitVec.ofNat_add_ofNat]
    congr 1
    ring
  | succ step ih =>
    rw [vectorPointer, ih]
    simp only [word, BitVec.ofNat_mul_ofNat, BitVec.ofNat_add_ofNat]
    congr 1
    ring

theorem output_modular {N M : Nat} (mem : KernelMemory N M) (c : KernelConfig)
    (pid lane : Nat) :
    outputPointer mem c pid lane = word (4 * (mem.outputBase + (pid * c.blockRows + lane))) := by
  simp only [outputPointer, row_modular, word, BitVec.ofNat_mul_ofNat, BitVec.ofNat_add_ofNat]
  congr 1
  ring

theorem row_small {N : Nat} (c : KernelConfig) (pid lane : Nat) (extent : N < 2^63)
    (program : pid < (N + c.blockRows - 1) / c.blockRows) (hl : lane < c.blockRows) :
    pid * c.blockRows + lane < 2^64 := by
  have bounds := config_bounds c
  have le := Nat.mul_le_mul_right c.blockRows (Nat.succ_le_iff.mpr program)
  have lim := Nat.div_mul_le_self (N + c.blockRows - 1) c.blockRows
  rw [Nat.succ_mul] at le
  omega

theorem column_small {M : Nat} (c : KernelConfig) (step lane : Nat) (extent : M < 2^63)
    (ht : step < chunks M c) (hl : lane ≤ c.blockColumns) :
    step * c.blockColumns + lane < 2^64 := by
  have bounds := config_bounds c
  have le := Nat.mul_le_mul_right c.blockColumns (Nat.succ_le_iff.mpr ht)
  have lim := Nat.div_mul_le_self (M + c.blockColumns - 1) c.blockColumns
  unfold chunks at le
  rw [Nat.succ_mul] at le
  omega

/-- Row masks also cover M=0, when no column body is reached. -/
theorem row_mask {x y : TensorMetadata} {c : KernelConfig} {pid : Nat}
    {mem : KernelMemory (rows x) (columns x)} (h : KernelContract x y c pid mem)
    (lane : Nat) (hl : lane < c.blockRows) :
    (rowWord c pid lane).toNat = pid * c.blockRows + lane ∧
    ((rowWord c pid lane).ult (word (rows x)) = Bool.true ↔ pid * c.blockRows + lane < rows x) := by
  have exactRow : (rowWord c pid lane).toNat = pid * c.blockRows + lane := by
    rw [row_modular]
    exact word_exact (row_small c pid lane h.inputs.extent.1 h.program hl)
  refine ⟨exactRow, ?_⟩
  rw [BitVec.ult_iff_toNat_lt, exactRow, word_exact (lt_trans h.inputs.extent.1 (by norm_num))]

/-- Column masks cover tail lanes; the vector mask is independent of row
activity, matching the source load. -/
theorem column_mask {M : Nat} (c : KernelConfig) (extent : M < 2^63)
    (step col : Nat) (ht : step < chunks M c) (hc : col < c.blockColumns) :
    (columnWord c step col).toNat = step * c.blockColumns + col ∧
    ((columnWord c step col).ult (word M) = Bool.true ↔ step * c.blockColumns + col < M) := by
  have exactCol : (columnWord c step col).toNat = step * c.blockColumns + col := by
    rw [column_modular]
    exact word_exact (column_small c step col extent ht (Nat.le_of_lt hc))
  refine ⟨exactCol, ?_⟩
  rw [BitVec.ult_iff_toNat_lt, exactCol, word_exact (lt_trans extent (by norm_num))]

/-- The unsigned while condition includes the initial zero and final backedge;
the loop cannot wrap before its mathematical stopping point. -/
theorem loop_mask {M : Nat} (c : KernelConfig) (extent : M < 2^63)
    (step : Nat) (ht : step ≤ chunks M c) :
    (word (step * c.blockColumns)).toNat = step * c.blockColumns ∧
    ((word (step * c.blockColumns)).ult (word M) = Bool.true ↔ step * c.blockColumns < M) := by
  have bounds := config_bounds c
  have le := Nat.mul_le_mul_right c.blockColumns ht
  have lim := Nat.div_mul_le_self (M + c.blockColumns - 1) c.blockColumns
  unfold chunks at le
  have he := word_exact (show step * c.blockColumns < 2^64 by omega)
  refine ⟨he, ?_⟩
  rw [BitVec.ult_iff_toNat_lt, he, word_exact (lt_trans extent (by norm_num))]

theorem matrix_active {N M : Nat} (mem : KernelMemory N M) (c : KernelConfig)
    (pid lane col step : Nat) (hr : pid * c.blockRows + lane < N)
    (hc : step * c.blockColumns + col < M) :
    (matrixPointer mem c pid lane col step).toNat =
      4 * (mem.matrixBase + (pid * c.blockRows + lane) * M + (step * c.blockColumns + col)) := by
  rw [matrix_modular]
  have bound := mem.matrixBound _ _ hr hc
  exact word_exact (by omega)

theorem vector_active {N M : Nat} (mem : KernelMemory N M) (c : KernelConfig)
    (col step : Nat) (hc : step * c.blockColumns + col < M) :
    (vectorPointer mem c col step).toNat = 4 * (mem.vectorBase + (step * c.blockColumns + col)) := by
  rw [vector_modular]
  have bound := mem.vectorBound _ hc
  exact word_exact (by omega)

theorem output_active {N M : Nat} (mem : KernelMemory N M) (c : KernelConfig)
    (pid lane : Nat) (hr : pid * c.blockRows + lane < N) :
    (outputPointer mem c pid lane).toNat = 4 * (mem.outputBase + (pid * c.blockRows + lane)) := by
  rw [output_modular]
  have bound := mem.outputBound _ hr
  exact word_exact (by omega)

/-- Output/input separation is at byte granularity, allowing A/B aliases. -/
theorem output_separate {N M : Nat} (mem : KernelMemory N M)
    (i r j b₁ b₂ : Nat) (hi : i < N) (hr : r < N) (hj : j < M)
    (h₁ : b₁ < 4) (h₂ : b₂ < 4) :
    4 * (mem.outputBase + i) + b₁ ≠ 4 * (mem.matrixBase + r * M + j) + b₂ ∧
    4 * (mem.outputBase + i) + b₁ ≠ 4 * (mem.vectorBase + j) + b₂ := by
  have hm := mem.matrixSeparate i r j hi hr hj
  have hv := mem.vectorSeparate i j hi hj
  omega

/-- Active offsets also fit the lengths of the contiguous source regions. -/
theorem region_bounds (N M i j : Nat) (hi : i < N) (hj : j < M) :
    i * M + j < N * M ∧ j < M ∧ i < N := by
  refine ⟨?_, hj, hi⟩
  calc
    i * M + j < i * M + M := Nat.add_lt_add_left hj _
    _ = (i + 1) * M := by ring
    _ ≤ N * M := Nat.mul_le_mul_right M (Nat.succ_le_iff.mpr hi)

/-- The source-access contract after widening: exact masks and loop guards,
active complete-word addresses, separate output bytes, and legal native IDs.
No host allocation or returned-tensor property is part of this judgment. -/
structure Correspondence (N M : Nat) (c : KernelConfig) (pid : Nat)
    (mem : KernelMemory N M) : Prop where
  scalars : ∀ n ∈ [N, M, 1], (scalarABI n).toNat = n
  iterations : ∀ step, step < chunks M c ↔ step * c.blockColumns < M
  nativeRows : ∀ lane, lane < c.blockRows → nativeRow c pid lane = rowWord c pid lane
  nativeColumns : ∀ step col, col < c.blockColumns → nativeColumn c step col = columnWord c step col
  rows : ∀ lane, lane < c.blockRows →
    (rowWord c pid lane).toNat = pid * c.blockRows + lane ∧
    ((rowWord c pid lane).ult (word N) = Bool.true ↔ pid * c.blockRows + lane < N)
  columns : ∀ step col, step < chunks M c → col < c.blockColumns →
    (columnWord c step col).toNat = step * c.blockColumns + col ∧
    ((columnWord c step col).ult (word M) = Bool.true ↔ step * c.blockColumns + col < M)
  loop : ∀ step, step ≤ chunks M c →
    (word (step * c.blockColumns)).toNat = step * c.blockColumns ∧
    ((word (step * c.blockColumns)).ult (word M) = Bool.true ↔ step * c.blockColumns < M)
  matrix : ∀ lane col step, pid * c.blockRows + lane < N → step * c.blockColumns + col < M →
    (matrixPointer mem c pid lane col step).toNat =
      4 * (mem.matrixBase + (pid * c.blockRows + lane) * M + (step * c.blockColumns + col))
  vector : ∀ col step, step * c.blockColumns + col < M →
    (vectorPointer mem c col step).toNat = 4 * (mem.vectorBase + (step * c.blockColumns + col))
  output : ∀ lane, pid * c.blockRows + lane < N →
    (outputPointer mem c pid lane).toNat = 4 * (mem.outputBase + (pid * c.blockRows + lane))
  separate : ∀ i r j b₁ b₂, i < N → r < N → j < M → b₁ < 4 → b₂ < 4 →
    4 * (mem.outputBase + i) + b₁ ≠ 4 * (mem.matrixBase + r * M + j) + b₂ ∧
    4 * (mem.outputBase + i) + b₁ ≠ 4 * (mem.vectorBase + j) + b₂

theorem correspondence {x y : TensorMetadata} {c : KernelConfig} {pid : Nat}
    {mem : KernelMemory (rows x) (columns x)} (h : KernelContract x y c pid mem) :
    Correspondence (rows x) (columns x) c pid mem := by
  refine ⟨?_, iterations (columns x) c, fun lane hl => native_row c pid lane h.programABI hl,
    native_column c, row_mask h, column_mask c h.inputs.extent.2,
    loop_mask c h.inputs.extent.2, matrix_active mem c pid,
    vector_active mem c, output_active mem c pid, output_separate mem⟩
  intro n member
  apply scalar_exact
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · exact h.inputs.extent.1
  · exact h.inputs.extent.2
  · norm_num

theorem matrix_actual_pointer {x y : TensorMetadata} {c : KernelConfig} {pid : Nat}
    {mem : KernelMemory (rows x) (columns x)} (h : KernelContract x y c pid mem)
    (lane col step : Nat) (hr : pid * c.blockRows + lane < rows x)
    (hc : step * c.blockColumns + col < columns x) :
    (matrixPointer mem c pid lane col step).toNat = x.dataPtr.toNat +
      4 * ((pid * c.blockRows + lane) * columns x + (step * c.blockColumns + col)) := by
  have nonempty : 0 < x.numel := by
    rw [matrix_numel x y h.inputs]
    exact Nat.mul_pos (by omega) (by omega)
  rw [matrix_active mem c pid lane col step hr hc, h.matrixPointer nonempty]
  ring

theorem vector_actual_pointer {x y : TensorMetadata} {c : KernelConfig} {pid : Nat}
    {mem : KernelMemory (rows x) (columns x)} (h : KernelContract x y c pid mem)
    (col step : Nat) (hc : step * c.blockColumns + col < columns x) :
    (vectorPointer mem c col step).toNat = y.dataPtr.toNat + 4 * (step * c.blockColumns + col) := by
  have nonempty : 0 < y.numel := by
    rw [vector_numel x y h.inputs]
    omega
  rw [vector_active mem c col step hc, h.vectorPointer nonempty]
  ring

/-- The active source pointers are the checked Python argument pointers plus
the canonical contiguous byte offset, not arbitrary symbolic region bases. -/
structure CheckedCorrespondence (x y : TensorMetadata) (c : KernelConfig) (pid : Nat)
    (mem : KernelMemory (VeriTile.Bench.Repairs.MatrixVectorMultip.rows x) (VeriTile.Bench.Repairs.MatrixVectorMultip.columns x)) : Prop extends
    Correspondence (VeriTile.Bench.Repairs.MatrixVectorMultip.rows x) (VeriTile.Bench.Repairs.MatrixVectorMultip.columns x) c pid mem where
  matrixPointer : ∀ lane col step, pid * c.blockRows + lane < VeriTile.Bench.Repairs.MatrixVectorMultip.rows x →
    step * c.blockColumns + col < VeriTile.Bench.Repairs.MatrixVectorMultip.columns x →
    (matrixPointer mem c pid lane col step).toNat = x.dataPtr.toNat +
      4 * ((pid * c.blockRows + lane) * VeriTile.Bench.Repairs.MatrixVectorMultip.columns x + (step * c.blockColumns + col))
  vectorPointer : ∀ col step, step * c.blockColumns + col < VeriTile.Bench.Repairs.MatrixVectorMultip.columns x →
    (vectorPointer mem c col step).toNat = y.dataPtr.toNat + 4 * (step * c.blockColumns + col)

theorem checked_correspondence {x y : TensorMetadata} {c : KernelConfig} {pid : Nat}
    {mem : KernelMemory (rows x) (columns x)} (h : KernelContract x y c pid mem) :
    CheckedCorrespondence x y c pid mem :=
  ⟨correspondence h, matrix_actual_pointer h, vector_actual_pointer h⟩

/-- Active original signed-i32 index counterexample and its exact repair. -/
theorem original_overflow :
    (BitVec.ofNat 32 2 * BitVec.ofNat 32 1073741824).toInt = -2147483648 := by decide

theorem repaired_offset : (word 2 * word 1073741824).toNat = 2147483648 := by decide

#axiomsClean row_mask
#axiomsClean column_mask
#axiomsClean loop_mask
#axiomsClean matrix_active
#axiomsClean vector_active
#axiomsClean output_active
#axiomsClean output_separate
#axiomsClean correspondence
#axiomsClean checked_correspondence
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.MatrixVectorMultip.KernelIndex
