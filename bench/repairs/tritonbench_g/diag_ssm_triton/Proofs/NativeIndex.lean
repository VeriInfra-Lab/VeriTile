import Mathlib.Tactic
import VeriTile.Meta.StatementAudit

/-! Signed-64 kernel intermediates. Bounds come from actual nonwrapping byte
allocations and signed tensor-size metadata, not a newly restricted shape set.
Grid ownership is mathematical; native runtime grid segmentation is separate. -/
set_option autoImplicit false
namespace VeriTile.Bench.Repairs.DiagSsmTriton.NativeIndex

def i64 (z : Int) : Int :=
  let q := z % (2 ^ 64 : Int)
  if q < 2 ^ 63 then q else q - 2 ^ 64

theorem i64_exact (z : Int) (lo : 0 ≤ z) (hi : z < 2 ^ 63) : i64 z = z := by
  have h64 : z < (2 ^ 64 : Int) := by omega
  unfold i64
  rw [Int.emod_eq_of_lt lo h64, if_pos hi]

def columns (batch dim : Nat) := batch * dim
def grid (n : Nat) := (n + 127) / 128
def column (pid lane : Nat) := pid * 128 + lane
def offset (n t pid lane : Nat) := t * n + column pid lane

theorem column_of_active (n pid lane : Nat) (hl : lane < 128)
    (ha : column pid lane < n) : pid < grid n := by
  simp only [column, grid] at *
  omega

theorem padded_column (n pid lane : Nat) (hp : pid < grid n) (hl : lane < 128) :
    column pid lane < n + 127 := by
  simp only [column, grid] at *
  omega

theorem column_owner (c : Nat) : column (c / 128) (c % 128) = c := by
  simpa [column, Nat.mul_comm] using Nat.div_add_mod c 128

theorem column_unique (p q i j : Nat) (hi : i < 128) (hj : j < 128)
    (h : column p i = column q j) : p = q ∧ i = j := by
  simp only [column] at h
  omega

theorem active_dimension (batch dim c : Nat) (h : c < columns batch dim) :
    0 < batch ∧ 0 < dim := by
  simp only [columns] at h
  constructor <;> nlinarith

theorem coordinate_owner (batch dim c : Nat) (h : c < columns batch dim) :
    c / dim < batch ∧ c % dim < dim ∧ (c / dim) * dim + c % dim = c := by
  have hd := (active_dimension batch dim c h).2
  exact ⟨(Nat.div_lt_iff_lt_mul hd).2 h, Nat.mod_lt _ hd,
    by simpa [Nat.mul_comm] using Nat.div_add_mod c dim⟩

theorem active_offset (length n t c : Nat) (ht : t < length) (hc : c < n) :
    t * n + c < length * n := by
  nlinarith

theorem temporal_unique (n t u c d : Nat) (hc : c < n) (hd : d < n)
    (h : t * n + c = u * n + d) : t = u ∧ c = d := by
  have hn : 0 < n := by omega
  rcases lt_trichotomy t u with htu | rfl | hut
  · have htu' := Nat.mul_le_mul_right n htu
    nlinarith
  · exact ⟨rfl, by omega⟩
  · have hut' := Nat.mul_le_mul_right n hut
    nlinarith

theorem component_unique (c d i j : Nat) (hi : i < 2) (hj : j < 2)
    (h : 2 * c + i = 2 * d + j) : c = d ∧ i = j := by omega

/-- Ordinary metadata of already materialized s and x allocations. Their
counts are required by the documented shapes. These are post-allocation
facts; failure to allocate is an API failure, not an input-domain cutoff. -/
structure AllocatedShape (length batch dim components : Nat) : Prop where
  component_count : components = 1 ∨ components = 2
  time_size : length < 2 ^ 63
  initial_bytes : components * 4 * (batch * dim) < 2 ^ 64
  sequence_bytes : components * 4 * (length * (batch * dim)) < 2 ^ 64

theorem allocated_counts {length batch dim components : Nat}
    (h : AllocatedShape length batch dim components) :
    components * (batch * dim) < 2 ^ 62 ∧
      components * (length * (batch * dim)) < 2 ^ 62 := by
  have hs := h.initial_bytes
  have hx := h.sequence_bytes
  constructor <;> nlinarith

theorem padded_static_i64 {length batch dim components : Nat}
    (h : AllocatedShape length batch dim components)
    (pid lane k : Nat) (hp : pid < grid (batch * dim)) (hl : lane < 128)
    (hk : k < components) :
    components * column pid lane + k < 2 ^ 63 := by
  have hc := padded_column (batch * dim) pid lane hp hl
  have hs := (allocated_counts h).1
  rcases h.component_count with he | he <;> subst components <;> norm_num at * <;> omega

theorem padded_time_i64 {length batch dim components : Nat}
    (h : AllocatedShape length batch dim components)
    (t pid lane k : Nat) (ht : t < length)
    (hp : pid < grid (batch * dim)) (hl : lane < 128) (hk : k < components) :
    components * offset (batch * dim) t pid lane + k < 2 ^ 63 := by
  have hc := padded_column (batch * dim) pid lane hp hl
  have hx := (allocated_counts h).2
  have hn : 0 < batch * dim := by
    by_contra hz
    have hz : batch * dim = 0 := by omega
    simp [grid, hz] at hp
  have htt : t * (batch * dim) + batch * dim ≤ length * (batch * dim) := by nlinarith
  unfold offset
  rcases h.component_count with he | he <;> subst components <;> norm_num at * <;> omega

theorem reverse_time (length i : Nat) (hi : i < length) :
    length - 1 - i < length ∧ (length : Int) - 1 - i = (length - 1 - i : Nat) := by
  omega

theorem previous_offset (n t c : Nat) (ht : 0 < t) :
    (t * n + c : Int) - n = ((t - 1) * n + c : Nat) := by
  have ht' : t = (t - 1) + 1 := by omega
  rw [ht']
  push_cast
  ring

/-- Each actual component of a live lane lies within the corresponding
tensor byte window. Padded inactive lanes never require a dereference. -/
theorem component_byte_end (count components index k : Nat)
    (hi : index < count) (hk : k < components) :
    4 * (components * index + k) + 4 ≤ 4 * (components * count) := by
  nlinarith

theorem loop_i64 {length batch dim components : Nat}
    (h : AllocatedShape length batch dim components) (i : Nat) (hi : i < length) :
    i64 length = length ∧ i64 i = i ∧ i64 ((length : Int) - 1 - i) = length - 1 - i ∧
      i64 ((i : Int) + 1) = i + 1 := by
  have hl : (length : Int) < 2 ^ 63 := by exact_mod_cast h.time_size
  have hii : (i : Int) < length := by exact_mod_cast hi
  exact ⟨i64_exact _ (by omega) hl, i64_exact _ (by omega) (by omega),
    i64_exact _ (by omega) (by omega), i64_exact _ (by omega) (by omega)⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.NativeIndex
