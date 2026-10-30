import bench.repairs.tritonbench_g.vector_addition_custom.Proofs.Rewrite

namespace VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous.Rewrite
open VeriTile.Triton FP.Structural FP.Guarded
noncomputable section

/-- The transformation domain cannot depend on either inactive load value. -/
theorem domain_independent_of_undef {α : Type} [Inhabited α]
    (M : Algebra α) (D : Domain α) (n B : Nat) (s : State α)
    (undef : (d : TileDType) → RegionName → Nat → Value α d) :
    (additionSite n B).Holds M D {s with undef := undef} ↔
      (additionSite n B).Holds M D s := by
  rw [site_values_iff, site_values_iff]
  constructor <;> intro h i hi <;> simpa [loadedValue, hi] using h i hi

theorem empty_domain {α : Type} [Inhabited α]
    (M : Algebra α) (D : Domain α) (B : Nat) (s : State α) :
    (additionSite 0 B).Holds M D s := by
  simp [site_values_iff]

/-- A one-lane tail constrains exactly the two loaded operands in that lane. -/
theorem singleton_tail_domain {α : Type} [Inhabited α]
    (M : Algebra α) (D : Domain α) (B : Nat) (s : State α)
    (hx : D .finite (M.fp32Load ((s.mem "x" (s.pids 0 * B)).read .real)))
    (hy : D .finite (M.fp32Load ((s.mem "y" (s.pids 0 * B)).read .real))) :
    (additionSite (s.pids 0 * B + 1) B).Holds M D s := by
  rw [site_values_iff]
  intro i hi
  have hi0 : i.1.val = 0 := by omega
  exact ⟨by simpa [loadedValue, hi0] using hx,
    by simpa [loadedValue, hi0] using hy⟩

theorem active_left_rejected {α : Type} [Inhabited α]
    (M : Algebra α) (D : Domain α) (n B : Nat) (s : State α)
    (i : TileIndex [B]) (hi : s.pids 0 * B + i.1.val < n)
    (hx : ¬ D .finite (loadedValue M "x" n B s i)) :
    ¬ (additionSite n B).Holds M D s := by
  intro h
  exact hx (((site_values_iff M D n B s).mp h i hi).1)

theorem active_right_rejected {α : Type} [Inhabited α]
    (M : Algebra α) (D : Domain α) (n B : Nat) (s : State α)
    (i : TileIndex [B]) (hi : s.pids 0 * B + i.1.val < n)
    (hy : ¬ D .finite (loadedValue M "y" n B s i)) :
    ¬ (additionSite n B).Holds M D s := by
  intro h
  exact hy (((site_values_iff M D n B s).mp h i hi).2)

#axiomsClean domain_independent_of_undef
#axiomsClean empty_domain
#axiomsClean singleton_tail_domain
#axiomsClean active_left_rejected
#axiomsClean active_right_rejected
end
end VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous.Rewrite
