import bench.repairs.tritonbench_g.add_value.Proofs.Rewrite

/-! Boundary checks for the observed-lane domain. Inactive load values remain
arbitrary; active nonfinite operands remain outside the existing rewrite domain. -/
namespace VeriTile.Bench.Repairs.AddValue.Contiguous.Rewrite
open VeriTile.Triton FP.Structural FP.Guarded
noncomputable section

theorem domain_independent_of_undef {α : Type} [Inhabited α]
    (M : Algebra α) (D : Domain α) (n : Nat) (s : State α)
    (undef : (d : TileDType) → RegionName → Nat → Value α d) :
    (additionSite n).Holds M D {s with undef := undef} ↔
      (additionSite n).Holds M D s := by
  rw [site_values_iff, site_values_iff]
  constructor <;> intro h i hi <;> simpa [loadedValue, hi] using h i hi

/-- No output lane is active, so no finite-value premise is needed. -/
theorem empty_domain {α : Type} [Inhabited α]
    (M : Algebra α) (D : Domain α) (s : State α) :
    (additionSite 0).Holds M D s := by
  simp [site_values_iff]

/-- Includes n=1/pid=0 and the one-element tail n=1025/pid=1. -/
theorem singleton_tail_domain {α : Type} [Inhabited α]
    (M : Algebra α) (D : Domain α) (s : State α)
    (hx : D .finite (M.fp32Load ((s.mem "x" (s.pids 0 * blockSize)).read .real)))
    (h10 : D .finite (M.literal (some .fp32) .real 10)) :
    (additionSite (s.pids 0 * blockSize + 1)).Holds M D s := by
  rw [site_values_iff]
  intro i hi
  have hi0 : i.1.val = 0 := by omega
  exact ⟨by simpa [loadedValue, hi0] using hx, h10⟩

/-- The complete memory-equivalence result needs no restriction on the
remaining 1023 lanes of this tail, even if their load values are nonfinite. -/
theorem singleton_tail_equiv {α : Type} [Inhabited α]
    (M : Algebra α) (D : Domain α)
    (hM : Models [] M D) (s : State α)
    (hx : D .finite (M.fp32Load ((s.mem "x" (s.pids 0 * blockSize)).read .real)))
    (h10 : D .finite (M.literal (some .fp32) .real 10)) :
    (originalExecution M (s.pids 0 * blockSize + 1) s).val.mem =
      (commutedExecution M (s.pids 0 * blockSize + 1) s).val.mem :=
  commuted_memory M D hM _ s (singleton_tail_domain M D s hx h10)

/-- With n=2/pid=0, lane 1 is active and its finiteness cannot be omitted. -/
theorem nonfinite_active_lane_rejected {α : Type} [Inhabited α]
    (M : Algebra α) (D : Domain α) (s : State α) (hp : s.pids 0 = 0)
    (hx : ¬ D .finite (M.fp32Load ((s.mem "x" 1).read .real))) :
    ¬ (additionSite 2).Holds M D s := by
  intro h
  have hf := (site_values_iff M D 2 s).mp h
    ((⟨1, by decide⟩ : Fin blockSize), PUnit.unit) (by simp [hp])
  exact hx (by simpa [loadedValue, hp] using hf.1)

#axiomsClean domain_independent_of_undef
#axiomsClean empty_domain
#axiomsClean singleton_tail_domain
#axiomsClean singleton_tail_equiv
#axiomsClean nonfinite_active_lane_rejected
end
end VeriTile.Bench.Repairs.AddValue.Contiguous.Rewrite
