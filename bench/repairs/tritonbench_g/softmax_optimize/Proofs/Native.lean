import bench.repairs.tritonbench_g.softmax_optimize.Proofs.Config
import Init.Data.BitVec.Lemmas

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.Native

/-- Bounds come from the actual successful FP32 allocation. They are not a
new bound on a Python shape or a certificate of a kernel execution. -/
structure Allocation (M N : Nat) where
  base : Nat
  bytes : Nat
  native_span : base + bytes ≤ 2 ^ 64
  holds_words : 4 * (M * N) ≤ bytes

def element (N row column : Nat) : Nat := row * N + column

theorem active_element_bound {M N : Nat} (row : Fin M) (column : Fin N) :
    element N row.val column.val < M * N := by
  unfold element
  calc
    row.val * N + column.val < row.val * N + N := Nat.add_lt_add_left column.isLt _
    _ = (row.val + 1) * N := by ring
    _ ≤ M * N := Nat.mul_le_mul_right N (by omega)

theorem allocated_word_bound {M N : Nat} (a : Allocation M N) :
    M * N ≤ 4611686018427387904 := by
  have h := a.native_span
  have hwords := a.holds_words
  norm_num at h
  omega

theorem allocated_column_bound {M N : Nat} (a : Allocation M N) (hM : 0 < M) :
    N ≤ 4611686018427387904 := by
  have h := allocated_word_bound a
  have hn : N ≤ M * N := by nlinarith
  omega

theorem active_byte_bound {M N : Nat} (a : Allocation M N)
    (row : Fin M) (column : Fin N) (byte : Fin 4) :
    a.base + 4 * element N row.val column.val + byte.val < a.base + a.bytes ∧
      a.base + 4 * element N row.val column.val + byte.val < 2 ^ 64 := by
  have h := active_element_bound row column
  have hw := a.holds_words
  have hb := byte.isLt
  have hlocal : a.base + 4 * element N row.val column.val + byte.val < a.base + a.bytes := by
    omega
  exact ⟨hlocal, lt_of_lt_of_le hlocal a.native_span⟩

/-- The source widens before the row product, column addition and pointer
scaling. The remaining global PID/launch compatibility is a separate port. -/
def PIDCompatible (M : Nat) : Prop := ∀ row : Fin M, row.val < 2 ^ 31

def pid32 (row : Nat) : Int := (BitVec.ofNat 32 row).toInt

theorem signed_exact (width value : Nat) (h : 2 * value < 2 ^ width) :
    (BitVec.ofNat width value).toInt = (value : Int) := by
  have hv : value < 2 ^ width := by omega
  have hm : (BitVec.ofNat width value).toNat = value := by
    simp [BitVec.toNat_ofNat, Nat.mod_eq_of_lt hv]
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hm]; exact h), hm]

theorem pid32_exact {M : Nat} (compatible : PIDCompatible M) (row : Fin M) :
    pid32 row.val = (row.val : Int) := by
  apply signed_exact
  have h := compatible row
  norm_num at h ⊢
  omega

def nativeElement (N row column : Nat) : Int :=
  ((BitVec.ofInt 64 (pid32 row) * BitVec.ofNat 64 N) + BitVec.ofNat 64 column).toInt

theorem widened_element_exact {M N : Nat} (a : Allocation M N)
    (compatible : PIDCompatible M) (row : Fin M) (column : Fin N) :
    nativeElement N row.val column.val = (element N row.val column.val : Int) := by
  unfold nativeElement
  rw [pid32_exact compatible row, BitVec.ofInt_natCast,
    ← BitVec.ofNat_mul, ← BitVec.ofNat_add]
  apply signed_exact
  have h := active_element_bound row column
  have htotal := allocated_word_bound a
  change 2 * element N row.val column.val < 2 ^ 64
  norm_num
  omega

/-- A full byte address includes the actual pointer base; no modulo equality
is substituted for an in-bounds equality. Every one of the four bytes is covered. -/
def nativeByte (base N row column byte : Nat) : Nat :=
  (BitVec.ofNat 64 base + BitVec.ofNat 64 (4 * element N row column + byte)).toNat

theorem native_byte_exact {M N : Nat} (a : Allocation M N)
    (row : Fin M) (column : Fin N) (byte : Fin 4) :
    nativeByte a.base N row.val column.val byte.val =
      a.base + 4 * element N row.val column.val + byte.val := by
  unfold nativeByte
  rw [← BitVec.ofNat_add, BitVec.toNat_ofNat]
  have h := (active_byte_bound a row column byte).2
  simpa only [Nat.add_assoc] using Nat.mod_eq_of_lt (show a.base + (4 * element N row.val column.val + byte.val) < 2 ^ 64 by omega)

/-- Even masked element offsets and the cdiv numerator are within signed i64
before pointer conversion. This follows from allocation metadata and tile cap. -/
theorem loop_intermediate_bound {M N : Nat} (a : Allocation M N) (hM : 0 < M)
    {row start lane : Nat} (hr : row < M) (hs : start < N) (hi : lane < Config.tile N) :
    N + Config.tile N - 1 < 2 ^ 63 ∧
      row * N + (start + lane) < 2 ^ 63 := by
  have hn := allocated_column_bound a hM
  have ht := Config.tile_le_cap N
  have total := allocated_word_bound a
  have hrow : (row + 1) * N ≤ M * N := Nat.mul_le_mul_right N (by omega)
  have hrn : row * N + N ≤ M * N := by nlinarith
  norm_num
  omega

/-- Contiguous metadata allows irrelevant singleton strides and a nonzero
storage offset. The proof uses logical coordinates rather than requiring a
new literal-stride restriction. -/
structure Dense2D (M N : Nat) where
  rowStride : Nat
  columnStride : Nat
  storageOffset : Nat
  row_dense : M ≤ 1 ∨ rowStride = N
  column_dense : N ≤ 1 ∨ columnStride = 1

def Dense2D.coordinate {M N : Nat} (layout : Dense2D M N) (row column : Nat) : Nat :=
  layout.storageOffset + row * layout.rowStride + column * layout.columnStride

theorem dense_coordinate {M N : Nat} (layout : Dense2D M N)
    (row : Fin M) (column : Fin N) :
    layout.coordinate row.val column.val = layout.storageOffset + element N row.val column.val := by
  have hr : row.val * layout.rowStride = row.val * N := by
    rcases layout.row_dense with h | h
    · have hz : row.val = 0 := by omega
      simp [hz]
    · rw [h]
  have hc : column.val * layout.columnStride = column.val := by
    rcases layout.column_dense with h | h
    · have hz : column.val = 0 := by omega
      simp [hz]
    · simp [h]
  simp only [Dense2D.coordinate, element, hr, hc, Nat.add_assoc]

def canonicalLayout (M N : Nat) : Dense2D M N :=
  ⟨N, 1, 0, Or.inr rfl, Or.inr rfl⟩

theorem nonempty_allocation_witness : Nonempty (Allocation 2 3) := by
  exact ⟨⟨4096, 27, by norm_num, by norm_num⟩⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SoftmaxOptimize.Native
