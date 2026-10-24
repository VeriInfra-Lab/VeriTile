import bench.repairs.tritonbench_g.triton_mul2.Proofs.Rewrite

/-! Boundary consequences of the observed-lane rewrite domain. No finite-value
condition is needed for masked load values or for an entirely empty program. -/
namespace VeriTile.Bench.Repairs.TritonMul2.Rewrite
open VeriTile.Triton FP.Structural FP.Guarded
noncomputable section

theorem domain_independent_of_undef {α : Type} [Inhabited α]
    (M : Algebra α) (D : Domain α) (X O : RegionName) (n B : Nat) (s : State α)
    (undef : (d : TileDType) → RegionName → Nat → Value α d) :
    (multiplicationSite X O n B).Holds M D {s with undef := undef} ↔
      (multiplicationSite X O n B).Holds M D s := by
  rw [site_values_iff, site_values_iff]
  constructor <;> intro h i hi <;> simpa [loadedValue, hi] using h i hi

theorem empty_domain {α : Type} [Inhabited α]
    (M : Algebra α) (D : Domain α) (X O : RegionName) (B : Nat) (s : State α) :
    (multiplicationSite X O 0 B).Holds M D s := by
  simp [site_values_iff]

/-- Includes a one-element input and the final one-element tail for any
supported block size; the other B-1 lanes have no numerical precondition. -/
theorem singleton_tail_domain {α : Type} [Inhabited α]
    (M : Algebra α) (D : Domain α) (X O : RegionName) (B : Nat) (s : State α)
    (hx : D .finite (M.fp32Load ((s.mem X (s.pids 0 * B)).read .real)))
    (h2 : D .finite (M.literal (some .fp32) .real 2)) :
    (multiplicationSite X O (s.pids 0 * B + 1) B).Holds M D s := by
  rw [site_values_iff]
  intro i hi
  have hi0 : i.1.val = 0 := by omega
  exact ⟨by simpa [loadedValue, hi0] using hx, h2⟩

theorem singleton_tail_equiv {α : Type} [Inhabited α]
    (M : Algebra α) (D : Domain α)
    (hM : Models [] M D) (X O : RegionName) (B : Nat) (s : State α)
    (hx : D .finite (M.fp32Load ((s.mem X (s.pids 0 * B)).read .real)))
    (h2 : D .finite (M.literal (some .fp32) .real 2)) :
    (originalExecution M X O (s.pids 0 * B + 1) B s).val.mem =
      (commutedExecution M X O (s.pids 0 * B + 1) B s).val.mem :=
  commuted_memory M D hM X O _ B s (singleton_tail_domain M D X O B s hx h2)

#axiomsClean domain_independent_of_undef
#axiomsClean empty_domain
#axiomsClean singleton_tail_domain
#axiomsClean singleton_tail_equiv
end
end VeriTile.Bench.Repairs.TritonMul2.Rewrite
