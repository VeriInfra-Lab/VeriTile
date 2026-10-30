import bench.tritonbench_g.layer_norm_liger.Proofs.BackwardModel
namespace VeriTile.Bench.Repairs.LayerNormLiger.Launch
open Finset

def rowsPerProgram (rows programs : Nat) := (rows+programs-1)/programs

theorem ceil_covers (rows programs : Nat) (hp : 0 < programs) :
    rows ≤ programs * rowsPerProgram rows programs := by
  exact (Nat.le_mul_iff_le_right hp).2 (Nat.le_refl _)

theorem ceil_positive (rows programs : Nat) (hr : 0 < rows) (hp : 0 < programs) :
    0 < rowsPerProgram rows programs := by
  have h := ceil_covers rows programs hp
  by_contra hz
  have : rowsPerProgram rows programs = 0 := by omega
  simp [this] at h
  omega

def owns (rows width pid row : Nat) := pid*width ≤ row ∧ row < min ((pid+1)*width) rows

theorem owns_iff (rows width pid row : Nat) (hw : 0 < width) :
    owns rows width pid row ↔ row < rows ∧ row/width = pid := by
  constructor
  · rintro ⟨lo,hi⟩
    exact ⟨lt_of_lt_of_le hi (Nat.min_le_right _ _),
      Nat.div_eq_of_lt_le lo (lt_of_lt_of_le hi (Nat.min_le_left _ _))⟩
  · rintro ⟨hr,rfl⟩
    refine ⟨Nat.div_mul_le_self _ _, lt_min ?_ hr⟩
    simpa [Nat.mul_comm] using Nat.lt_mul_div_succ row hw

theorem exactly_one_owner (rows programs row : Nat) (hp : 0 < programs) (hr : row < rows) :
    ∃! pid, pid < programs ∧ owns rows (rowsPerProgram rows programs) pid row := by
  have hw := ceil_positive rows programs (by omega) hp
  have hb := ceil_covers rows programs hp
  refine ⟨row/rowsPerProgram rows programs, ⟨?_,?_⟩,?_⟩
  · exact (Nat.div_lt_iff_lt_mul hw).2 (lt_of_lt_of_le hr hb)
  · exact (owns_iff _ _ _ _ hw).2 ⟨hr,rfl⟩
  · intro pid h
    exact ((owns_iff _ _ _ _ hw).1 h.2).2.symm

/-- The source's contiguous group interval is the fiber of integer row ownership. -/
theorem group_interval (rows width pid : Nat) (hw : 0 < width) :
    Ico (pid*width) (min ((pid+1)*width) rows) = (range rows).filter (fun row => row/width=pid) := by
  ext row
  simp only [mem_Ico, mem_filter, mem_range]
  exact owns_iff rows width pid row hw

/-- Summing all DW/DB group partials covers each logical input row once. -/
theorem group_sum (rows programs : Nat) (hp : 0 < programs) (hr : 0 < rows) (f : Nat → ℝ) :
    (∑ pid ∈ range programs, ∑ row ∈ Ico (pid*rowsPerProgram rows programs)
      (min ((pid+1)*rowsPerProgram rows programs) rows), f row) = ∑ row ∈ range rows, f row := by
  have hw := ceil_positive rows programs hr hp
  simp_rw [group_interval rows _ _ hw]
  apply Finset.sum_fiberwise_of_maps_to
  intro row hrow
  simp only [mem_range] at hrow ⊢
  exact (Nat.div_lt_iff_lt_mul hw).2 (lt_of_lt_of_le hrow (ceil_covers rows programs hp))

/-- Contiguous row-major output cells have unique row/column ownership. -/
theorem row_major_injective {columns r s i j : Nat} (hi : i < columns) (hj : j < columns)
    (h : r*columns+i=s*columns+j) : r=s ∧ i=j := by
  have hr : (r*columns+i)/columns=r := Nat.div_eq_of_lt_le (by omega) (by nlinarith)
  have hs : (s*columns+j)/columns=s := Nat.div_eq_of_lt_le (by omega) (by nlinarith)
  have hrs : r=s := by rw [← hr, ← hs, h]
  exact ⟨hrs, by subst s; omega⟩

#axiomsClean ceil_covers
#axiomsClean exactly_one_owner
#axiomsClean group_interval
#axiomsClean group_sum
#axiomsClean row_major_injective
end VeriTile.Bench.Repairs.LayerNormLiger.Launch
