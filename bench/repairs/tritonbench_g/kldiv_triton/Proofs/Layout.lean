import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Math
import Mathlib.Tactic

namespace VeriTile.Bench.Repairs.KldivTriton.Layout
set_option autoImplicit false
set_option maxHeartbeats 2000000

/-- The exact nonnegative-stride guard in `_check_writable_matrix`.
Torch's strided tensors have nonnegative strides; logical negative views are
resolved by the coordinate copy contract, not by negative physical strides. -/
def Writable (rows cols sr sc : Nat) : Prop :=
  rows = 0 ∨ cols = 0 ∨
    ((1 < rows → 0 < sr) ∧ (1 < cols → 0 < sc) ∧
      (1 < rows → 1 < cols →
        rows ≤ sc / Nat.gcd sr sc ∨ cols ≤ sr / Nat.gcd sr sc))

def index (offset sr sc r c : Nat) : Nat := offset + r * sr + c * sc

private theorem separated (rows cols sr sc r₁ r₂ c₁ c₂ : Nat)
    (hsr : 0 < sr) (hsc : 0 < sc)
    (hr₁ : r₁ < rows) (hr₂ : r₂ < rows) (hc₁ : c₁ < cols) (hc₂ : c₂ < cols)
    (guard : rows ≤ sc / Nat.gcd sr sc ∨ cols ≤ sr / Nat.gcd sr sc)
    (ordered : r₁ < r₂) : r₁ * sr + c₁ * sc ≠ r₂ * sr + c₂ * sc := by
  intro collision
  have reverse : c₂ < c₁ := by
    by_contra h
    have := Nat.mul_le_mul_right sc (show c₁ ≤ c₂ by omega)
    nlinarith
  have hd : (r₂ - r₁) * sr = (c₁ - c₂) * sc := by
    have h₁ := Nat.sub_add_cancel ordered.le
    have h₂ := Nat.sub_add_cancel reverse.le
    nlinarith
  have hg : 0 < Nat.gcd sr sc := Nat.gcd_pos_of_pos_left sc hsr
  have cancel : (r₂ - r₁) * (sr / Nat.gcd sr sc) =
      (c₁ - c₂) * (sc / Nat.gcd sr sc) := by
    apply Nat.eq_of_mul_eq_mul_right hg
    simpa only [Nat.mul_assoc, Nat.div_mul_cancel (Nat.gcd_dvd_left sr sc),
      Nat.div_mul_cancel (Nat.gcd_dvd_right sr sc)] using hd
  have coprime := Nat.coprime_div_gcd_div_gcd hg
  have dr : sc / Nat.gcd sr sc ∣ r₂ - r₁ := by
    apply coprime.symm.dvd_of_dvd_mul_right
    rw [cancel]
    exact dvd_mul_left _ _
  have dc : sr / Nat.gcd sr sc ∣ c₁ - c₂ := by
    apply coprime.dvd_of_dvd_mul_right
    rw [← cancel]
    exact dvd_mul_left _ _
  have lr := Nat.le_of_dvd (show 0 < r₂ - r₁ by omega) dr
  have lc := Nat.le_of_dvd (show 0 < c₁ - c₂ by omega) dc
  rcases guard with h | h <;> omega

/-- Passing the source's gcd test makes every logical output coordinate distinct.
There is no row-size, stride-size, or contiguous-layout restriction here. -/
theorem injective (rows cols offset sr sc : Nat) (guard : Writable rows cols sr sc)
    (r₁ r₂ : Fin rows) (c₁ c₂ : Fin cols)
    (collision : index offset sr sc r₁.val c₁.val = index offset sr sc r₂.val c₂.val) :
    r₁ = r₂ ∧ c₁ = c₂ := by
  rcases guard with empty | empty | ⟨hrows, hcols, bound⟩
  · have := r₁.isLt; omega
  · have := c₁.isLt; omega
  have plain : r₁.val * sr + c₁.val * sc = r₂.val * sr + c₂.val * sc := by
    unfold index at collision
    omega
  by_cases hr : r₁ = r₂
  · refine ⟨hr, ?_⟩
    by_cases hc : 1 < cols
    · apply Fin.ext
      apply Nat.eq_of_mul_eq_mul_right (hcols hc)
      simpa only [hr, Nat.add_left_cancel_iff] using plain
    · apply Fin.ext
      have := c₁.isLt
      have := c₂.isLt
      omega
  · have hrn : r₁.val ≠ r₂.val := fun h => hr (Fin.ext h)
    have hrows' : 1 < rows := by have := r₁.isLt; have := r₂.isLt; omega
    have hcols' : 1 < cols := by
      by_contra h
      have hc : c₁.val = c₂.val := by have := c₁.isLt; have := c₂.isLt; omega
      rw [hc, Nat.add_right_cancel_iff] at plain
      exact hrn (Nat.eq_of_mul_eq_mul_right (hrows hrows') plain)
    have hguard := bound hrows' hcols'
    rcases lt_or_gt_of_ne hrn with order | order
    · exact False.elim (separated rows cols sr sc r₁.val r₂.val c₁.val c₂.val
        (hrows hrows') (hcols hcols') r₁.isLt r₂.isLt c₁.isLt c₂.isLt hguard order plain)
    · exact False.elim (separated rows cols sr sc r₂.val r₁.val c₂.val c₁.val
        (hrows hrows') (hcols hcols') r₂.isLt r₁.isLt c₂.isLt c₁.isLt hguard order plain.symm)

#axiomsClean injective
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.KldivTriton.Layout
