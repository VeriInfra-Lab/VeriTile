import bench.tritonbench_g.matrix_vector_multip.Proofs.WholeFinite

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.Index
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeHost WholeGrid
noncomputable section

abbrev word (n : Nat) : BitVec 64 := BitVec.ofNat 64 n

/-- The ordinary Triton JIT integer ABI for nonnegative Torch metadata:
1 may be constexpr, other signed32 values use i32, and larger values use i64.
`tl.cast(..., tl.uint64)` sign-extends i32 before any arithmetic. -/
def scalarABI (n : Nat) : BitVec 64 :=
  if n = 1 then word 1 else if n < 2^31 then
    (BitVec.ofNat 32 n).signExtend 64 else word n

def programIdABI (pid : Nat) : BitVec 64 := (BitVec.ofNat 32 pid).signExtend 64

theorem i32_to_u64 (n : Nat) (small : n < 2^31) :
    (BitVec.ofNat 32 n).signExtend 64 = word n := by
  change BitVec.ofInt 64 (WholeFinite.i32 n).toInt = word n
  rw [WholeFinite.i32_exact small]
  exact BitVec.ofInt_natCast 64 n

theorem scalarABI_word (n : Nat) : scalarABI n = word n := by
  unfold scalarABI
  split
  · rename_i one
    rw [one]
  · split
    · rename_i small
      exact i32_to_u64 n small
    · rfl

theorem programIdABI_word (pid : Nat) (hardware : pid < 2^31) :
    programIdABI pid = word pid := i32_to_u64 pid hardware

def row (c : Config) (pid lane : Nat) : BitVec 64 :=
  word pid * word c.blockRows + word lane

def column (c : Config) (step lane : Nat) : BitVec 64 :=
  word step * word c.blockColumns + word lane

/-- `addptr` may interpret an offset as signed64, but pointer-byte arithmetic
uses its same 64-bit residue. This holds even for masked huge-stride offsets. -/
theorem signed_pointer_offset (offset : BitVec 64) :
    BitVec.ofInt 64 (4*offset.toInt) = word 4 * offset := by
  rw [BitVec.ofInt_mul, BitVec.ofInt_toInt]
  rfl

def nativeColumn (c : Config) (step lane : Nat) : BitVec 64 :=
  word step * word c.blockColumns + (BitVec.ofNat 32 lane).signExtend 64

theorem nativeColumn_word (c : Config) (step lane : Nat)
    (hl : lane < c.blockColumns) : nativeColumn c step lane = column c step lane := by
  have size : c.blockColumns ≤ 128 := by
    have h := c.columns_mem
    simp only [List.mem_cons, List.not_mem_nil, or_false] at h
    omega
  simp only [nativeColumn, column, i32_to_u64 lane (by omega)]

/-- Actual unsigned64 pointer rebinding, including masked lanes and backedges.
Intermediate products may wrap; no inactive-lane address bound is imposed. -/
def matrixPointer (x : Inputs) (c : Config) (pid lane col : Nat) : Nat → BitVec 64
  | 0 => word (4*x.matrixBase) + word 4 *
      (row c pid lane * word x.strideRow + word col * word x.strideColumn)
  | step+1 => matrixPointer x c pid lane col step +
      word 4 * (word c.blockColumns * word x.strideColumn)

def vectorPointer (x : Inputs) (c : Config) (col : Nat) : Nat → BitVec 64
  | 0 => word (4*x.vectorBase) + word 4 * (word col * word x.strideVector)
  | step+1 => vectorPointer x c col step +
      word 4 * (word c.blockColumns * word x.strideVector)

def outputPointer (base : Nat) (c : Config) (pid lane : Nat) : BitVec 64 :=
  word (4*base) + word 4 * row c pid lane

/-- These are tensor metadata and backing-allocation facts, not the old
signed32 product restrictions. All legal nonnegative strides, including huge
singleton strides and zero strides, remain allowed. -/
structure TensorBounds {α : Type} {entry : Nat → α} {protectedCells : Nat → Prop}
    (x : Inputs) (a : Allocated α x entry protectedCells) : Prop where
  rows : x.rows < 2^63
  columns : x.columns < 2^63
  strides : ∀ value ∈ [x.strideRow, x.strideColumn, x.strideVector], value < 2^63
  matrix : ∀ i j, i < x.rows → j < x.columns → 4*x.matrixAddress i j+4 ≤ 2^64
  vector : ∀ j, j < x.vectorLength → 4*x.vectorAddress j+4 ≤ 2^64
  output : ∀ i, i < x.rows → 4*(a.base+i)+4 ≤ 2^64

/-- Successful CUDA launch supplies a representable program id for every
actually executed tuning/final trial. This is ordinary launch validity, not a
new bound on accepted input shapes: launch failures remain explicit outcomes. -/
structure Ready {α : Type} {entry : Nat → α} {cells : Nat → Prop}
    (x : Inputs) (a : Allocated α x entry cells) (plan : Plan x) : Prop where
  tensor : TensorBounds x a
  launchable : ∀ t, t ∈ plan.trials ∨ t = plan.selected →
    gridSize (x.parameters t.config) < 2^31

def nativeRow (c : Config) (pid lane : Nat) : BitVec 64 :=
  programIdABI pid * word c.blockRows + (BitVec.ofNat 32 lane).signExtend 64

theorem nativeRow_word (c : Config) (pid lane : Nat)
    (hardware : pid < 2^31) (hl : lane < c.blockRows) :
    nativeRow c pid lane = row c pid lane := by
  have size : c.blockRows ≤ 8 := by
    have h := c.rows_mem
    simp only [List.mem_cons, List.not_mem_nil, or_false] at h
    omega
  simp only [nativeRow, row, programIdABI_word pid hardware,
    i32_to_u64 lane (by omega)]

def PlanCorrespondence (x : Inputs) (plan : Plan x) : Prop :=
  ∀ t, t ∈ plan.trials ∨ t = plan.selected → ∀ pid lane,
    pid < gridSize (x.parameters t.config) → lane < t.config.blockRows →
    nativeRow t.config pid lane = row t.config pid lane

theorem plan_correspondence {α : Type} {x : Inputs} {entry : Nat → α}
    {cells : Nat → Prop} {a : Allocated α x entry cells} {plan : Plan x}
    (ready : Ready x a plan) : PlanCorrespondence x plan := by
  intro t used pid lane hp hl
  exact nativeRow_word t.config pid lane (lt_trans hp (ready.launchable t used)) hl

theorem row_word (c : Config) (pid lane : Nat) :
    row c pid lane = word (pid*c.blockRows+lane) := by
  simp only [row, word, BitVec.ofNat_mul_ofNat, BitVec.ofNat_add_ofNat]

theorem column_word (c : Config) (step lane : Nat) :
    column c step lane = word (step*c.blockColumns+lane) := by
  simp only [column, word, BitVec.ofNat_mul_ofNat, BitVec.ofNat_add_ofNat]

theorem matrix_modular (x : Inputs) (c : Config) (pid lane col step : Nat) :
    matrixPointer x c pid lane col step = word
      (4*(x.matrixBase+(pid*c.blockRows+lane)*x.strideRow+
        (step*c.blockColumns+col)*x.strideColumn)) := by
  induction step with
  | zero =>
    simp only [matrixPointer, row_word, word, BitVec.ofNat_mul_ofNat,
      BitVec.ofNat_add_ofNat]
    congr 1
    ring
  | succ step ih =>
    rw [matrixPointer, ih]
    simp only [word, BitVec.ofNat_mul_ofNat, BitVec.ofNat_add_ofNat]
    congr 1
    ring

theorem vector_modular (x : Inputs) (c : Config) (col step : Nat) :
    vectorPointer x c col step = word
      (4*(x.vectorBase+(step*c.blockColumns+col)*x.strideVector)) := by
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

theorem output_modular (base : Nat) (c : Config) (pid lane : Nat) :
    outputPointer base c pid lane = word (4*(base+pid*c.blockRows+lane)) := by
  simp only [outputPointer, row_word, word, BitVec.ofNat_mul_ofNat, BitVec.ofNat_add_ofNat]
  congr 1
  ring

theorem word_exact {n : Nat} (bound : n < 2^64) : (word n).toNat = n := by
  rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt bound]

/-- Every scalar metadata argument reaches the uint64 arithmetic model with
its exact nonnegative value, covering i32/i64 and constexpr-one specialization. -/
theorem scalarABI_exact (n : Nat) (metadata : n < 2^63) : (scalarABI n).toNat = n := by
  rw [scalarABI_word]
  exact word_exact (lt_trans metadata (by norm_num))

theorem metadata_correspondence {α : Type} {x : Inputs} {entry : Nat → α}
    {cells : Nat → Prop} (a : Allocated α x entry cells) (bounds : TensorBounds x a) :
    ∀ value ∈ [x.rows, x.columns, x.strideRow, x.strideColumn, x.strideVector, 1],
      (scalarABI value).toNat = value := by
  intro value member
  apply scalarABI_exact
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with h | h | h | h | h | h
  · subst value; exact bounds.rows
  · subst value; exact bounds.columns
  · subst value; exact bounds.strides _ (by simp)
  · subst value; exact bounds.strides _ (by simp)
  · subst value; exact bounds.strides _ (by simp)
  · subst value; norm_num

theorem row_small (x : Inputs) (c : Config) (pid lane : Nat)
    (rows : x.rows < 2^63) (hp : pid < gridSize (x.parameters c))
    (hl : lane < c.blockRows) : pid*c.blockRows+lane < 2^64 := by
  have padded := WholeFinite.padded_row x c pid lane hp hl
  have limit := Nat.div_mul_le_self (x.rows+c.blockRows-1) c.blockRows
  have size : c.blockRows ≤ 8 := by
    have h := c.rows_mem
    simp only [List.mem_cons, List.not_mem_nil, or_false] at h
    omega
  change pid*c.blockRows+lane <
    ((x.rows+c.blockRows-1)/c.blockRows)*c.blockRows at padded
  omega

theorem column_small (x : Inputs) (c : Config) (step lane : Nat)
    (columns : x.columns < 2^63) (ht : step < WholeFinite.chunks x c)
    (hl : lane ≤ c.blockColumns) : step*c.blockColumns+lane < 2^64 := by
  have size : c.blockColumns ≤ 128 := by
    have h := c.columns_mem
    simp only [List.mem_cons, List.not_mem_nil, or_false] at h
    omega
  have le := Nat.mul_le_mul_right c.blockColumns
    (show step+1 ≤ WholeFinite.chunks x c by omega)
  have limit := Nat.div_mul_le_self (x.columns+c.blockColumns-1) c.blockColumns
  change (step+1)*c.blockColumns ≤
    ((x.columns+c.blockColumns-1)/c.blockColumns)*c.blockColumns at le
  rw [Nat.add_mul, Nat.one_mul] at le
  omega

/-- Unsigned comparisons cover padded tails and the final while backedge,
including M near INT64_MAX. The candidate uses `while`, not signed scf.for. -/
theorem masks {α : Type} {x : Inputs} {entry : Nat → α} {protectedCells : Nat → Prop}
    (a : Allocated α x entry protectedCells) (bounds : TensorBounds x a)
    (c : Config) (pid step lane col : Nat)
    (hp : pid < gridSize (x.parameters c)) (ht : step < WholeFinite.chunks x c)
    (hl : lane < c.blockRows) (hc : col < c.blockColumns) :
    (row c pid lane).toNat = pid*c.blockRows+lane ∧
    (column c step col).toNat = step*c.blockColumns+col ∧
    (column c step c.blockColumns).toNat = step*c.blockColumns+c.blockColumns ∧
    ((row c pid lane).ult (word x.rows) = Bool.true ↔ pid*c.blockRows+lane < x.rows) ∧
    ((column c step col).ult (word x.columns) = Bool.true ↔ step*c.blockColumns+col < x.columns) := by
  have hr := word_exact (row_small x c pid lane bounds.rows hp hl)
  have hc' := word_exact (column_small x c step col bounds.columns ht (Nat.le_of_lt hc))
  have he := word_exact (column_small x c step c.blockColumns bounds.columns ht (by rfl))
  rw [←row_word] at hr
  rw [←column_word] at hc' he
  refine ⟨hr, hc', he, ?_, ?_⟩
  · rw [BitVec.ult_iff_toNat_lt, hr, word_exact (lt_trans bounds.rows (by norm_num))]
  · rw [BitVec.ult_iff_toNat_lt, hc', word_exact (lt_trans bounds.columns (by norm_num))]

/-- Every actually accessed complete word has the ordinary tensor coordinate;
all other formed pointers are allowed to wrap and remain masked. -/
theorem active {α : Type} {x : Inputs} {entry : Nat → α} {protectedCells : Nat → Prop}
    (a : Allocated α x entry protectedCells) (bounds : TensorBounds x a)
    (compatible : x.columns = x.vectorLength) (c : Config) (pid step lane col : Nat)
    (hi : pid*c.blockRows+lane < x.rows) (hj : step*c.blockColumns+col < x.columns) :
    (matrixPointer x c pid lane col step).toNat =
      4*x.matrixAddress (pid*c.blockRows+lane) (step*c.blockColumns+col) ∧
    (vectorPointer x c col step).toNat = 4*x.vectorAddress (step*c.blockColumns+col) ∧
    (outputPointer a.base c pid lane).toNat = 4*(a.base+(pid*c.blockRows+lane)) := by
  rw [matrix_modular, vector_modular, output_modular]
  have hm := bounds.matrix _ _ hi hj
  have hv := bounds.vector _ (by simpa only [←compatible] using hj)
  have ho := bounds.output _ hi
  refine ⟨word_exact (by simpa [Inputs.matrixAddress] using (show 4*x.matrixAddress
    (pid*c.blockRows+lane) (step*c.blockColumns+col) < 2^64 by omega)),
    word_exact (by simpa [Inputs.vectorAddress] using (show 4*x.vectorAddress
      (step*c.blockColumns+col) < 2^64 by omega)), ?_⟩
  simpa only [Nat.add_assoc] using word_exact (show 4*(a.base+(pid*c.blockRows+lane)) < 2^64 by omega)

/-- Output stores also exist when the reduction extent is zero. -/
theorem output_active {α : Type} {x : Inputs} {entry : Nat → α} {cells : Nat → Prop}
    (a : Allocated α x entry cells) (bounds : TensorBounds x a)
    (c : Config) (pid lane : Nat) (hi : pid*c.blockRows+lane < x.rows) :
    (outputPointer a.base c pid lane).toNat = 4*(a.base+(pid*c.blockRows+lane)) := by
  rw [output_modular]
  have h := bounds.output _ hi
  simpa only [Nat.add_assoc] using word_exact (show 4*(a.base+(pid*c.blockRows+lane)) < 2^64 by omega)

/-- Vector loads use only the column mask, even for an inactive matrix row. -/
theorem vector_active {α : Type} {x : Inputs} {entry : Nat → α} {cells : Nat → Prop}
    (a : Allocated α x entry cells) (bounds : TensorBounds x a)
    (c : Config) (step col : Nat) (hj : step*c.blockColumns+col < x.vectorLength) :
    (vectorPointer x c col step).toNat = 4*x.vectorAddress (step*c.blockColumns+col) := by
  rw [vector_modular]
  have h := bounds.vector _ hj
  exact word_exact (by simpa [Inputs.vectorAddress] using
    (show 4*x.vectorAddress (step*c.blockColumns+col) < 2^64 by omega))

/-- Row masks are checked even when M=0 and no loop body is executed. -/
theorem row_correspondence {α : Type} {x : Inputs} {entry : Nat → α} {cells : Nat → Prop}
    (a : Allocated α x entry cells) (bounds : TensorBounds x a)
    (c : Config) (pid lane : Nat) (hp : pid < gridSize (x.parameters c))
    (hl : lane < c.blockRows) :
    (row c pid lane).toNat = pid*c.blockRows+lane ∧
    ((row c pid lane).ult (word x.rows) = Bool.true ↔ pid*c.blockRows+lane < x.rows) := by
  have exactRow : (row c pid lane).toNat = pid*c.blockRows+lane := by
    rw [row_word]
    exact word_exact (row_small x c pid lane bounds.rows hp hl)
  refine ⟨exactRow, ?_⟩
  rw [BitVec.ult_iff_toNat_lt, exactRow, word_exact (lt_trans bounds.rows (by norm_num))]

/-- Every reached while condition, including the initial zero and the final
backedge, has the natural-number loop meaning for all tensor metadata sizes. -/
theorem loop_correspondence {α : Type} {x : Inputs} {entry : Nat → α} {cells : Nat → Prop}
    (a : Allocated α x entry cells) (bounds : TensorBounds x a)
    (c : Config) (step : Nat) (ht : step ≤ WholeFinite.chunks x c) :
    (word (step*c.blockColumns)).toNat = step*c.blockColumns ∧
    ((word (step*c.blockColumns)).ult (word x.columns) = Bool.true ↔ step*c.blockColumns < x.columns) := by
  have size : c.blockColumns ≤ 128 := by
    have h := c.columns_mem
    simp only [List.mem_cons, List.not_mem_nil, or_false] at h
    omega
  have le := Nat.mul_le_mul_right c.blockColumns ht
  have limit := Nat.div_mul_le_self (x.columns+c.blockColumns-1) c.blockColumns
  change step*c.blockColumns ≤ ((x.columns+c.blockColumns-1)/c.blockColumns)*c.blockColumns at le
  have exactLoop := word_exact (show step*c.blockColumns < 2^64 by have h := bounds.columns; omega)
  refine ⟨exactLoop, ?_⟩
  rw [BitVec.ult_iff_toNat_lt, exactLoop, word_exact (lt_trans bounds.columns (by norm_num))]

/-- Full finite-width statement. Inactive pointers retain modular arithmetic;
only accessed words need ordinary backing-allocation bounds. -/
structure Correspondence {α : Type} {x : Inputs} {entry : Nat → α} {cells : Nat → Prop}
    (a : Allocated α x entry cells) : Prop where
  matrixModular : ∀ c pid lane col step, matrixPointer x c pid lane col step = word
    (4*(x.matrixBase+(pid*c.blockRows+lane)*x.strideRow+(step*c.blockColumns+col)*x.strideColumn))
  vectorModular : ∀ c col step, vectorPointer x c col step = word
    (4*(x.vectorBase+(step*c.blockColumns+col)*x.strideVector))
  outputModular : ∀ c pid lane, outputPointer a.base c pid lane = word
    (4*(a.base+pid*c.blockRows+lane))
  masks : ∀ c pid step lane col, pid < gridSize (x.parameters c) →
    step < WholeFinite.chunks x c → lane < c.blockRows → col < c.blockColumns →
    (row c pid lane).toNat = pid*c.blockRows+lane ∧
    (column c step col).toNat = step*c.blockColumns+col ∧
    (column c step c.blockColumns).toNat = step*c.blockColumns+c.blockColumns ∧
    ((row c pid lane).ult (word x.rows) = Bool.true ↔ pid*c.blockRows+lane < x.rows) ∧
    ((column c step col).ult (word x.columns) = Bool.true ↔ step*c.blockColumns+col < x.columns)
  active : x.columns = x.vectorLength → ∀ c pid step lane col,
    pid*c.blockRows+lane < x.rows → step*c.blockColumns+col < x.columns →
    (matrixPointer x c pid lane col step).toNat =
      4*x.matrixAddress (pid*c.blockRows+lane) (step*c.blockColumns+col) ∧
    (vectorPointer x c col step).toNat = 4*x.vectorAddress (step*c.blockColumns+col) ∧
    (outputPointer a.base c pid lane).toNat = 4*(a.base+(pid*c.blockRows+lane))
  outputActive : ∀ c pid lane, pid*c.blockRows+lane < x.rows →
    (outputPointer a.base c pid lane).toNat = 4*(a.base+(pid*c.blockRows+lane))
  vectorActive : ∀ c step col, step*c.blockColumns+col < x.vectorLength →
    (vectorPointer x c col step).toNat = 4*x.vectorAddress (step*c.blockColumns+col)
  rowMask : ∀ c pid lane, pid < gridSize (x.parameters c) → lane < c.blockRows →
    (row c pid lane).toNat = pid*c.blockRows+lane ∧
    ((row c pid lane).ult (word x.rows) = Bool.true ↔ pid*c.blockRows+lane < x.rows)
  loop : ∀ c step, step ≤ WholeFinite.chunks x c →
    (word (step*c.blockColumns)).toNat = step*c.blockColumns ∧
    ((word (step*c.blockColumns)).ult (word x.columns) = Bool.true ↔ step*c.blockColumns < x.columns)
  pointerOffset : ∀ offset : BitVec 64,
    BitVec.ofInt 64 (4*offset.toInt) = word 4 * offset
  nativeColumns : ∀ c step lane, lane < c.blockColumns →
    nativeColumn c step lane = column c step lane
  metadata : ∀ value ∈ [x.rows, x.columns, x.strideRow, x.strideColumn, x.strideVector, 1],
    (scalarABI value).toNat = value
  freshBytes : ∀ i addr b₁ b₂, i < x.rows → cells addr → b₁ < 4 → b₂ < 4 →
    4*(a.base+i)+b₁ ≠ 4*addr+b₂

theorem correspondence {α : Type} {x : Inputs} {entry : Nat → α} {cells : Nat → Prop}
    (a : Allocated α x entry cells) (bounds : TensorBounds x a) : Correspondence a :=
  ⟨matrix_modular x, vector_modular x, output_modular a.base, masks a bounds,
    active a bounds, output_active a bounds, vector_active a bounds,
    row_correspondence a bounds, loop_correspondence a bounds, signed_pointer_offset,
    nativeColumn_word, metadata_correspondence a bounds,
    WholeFinite.output_bytes_separate a⟩

/-- The original source has an active contiguous-layout counterexample:
N=3, M=strideRow=2^30, row=2. Both scalar inputs fit signed32 but their
product points before A. The revised domain still admits these dimensions. -/
theorem original_contiguous_overflow :
    ((BitVec.ofNat 32 2 * BitVec.ofNat 32 1073741824).toInt = -2147483648) := by decide

/-- The former signed32 failure is now exactly representable. -/
theorem former_overflow_repaired :
    ((word 2 * word 1073741824).toNat = 2147483648) := by decide

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.Index
