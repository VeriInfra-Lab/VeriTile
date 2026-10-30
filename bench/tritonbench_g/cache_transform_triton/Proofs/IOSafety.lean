import bench.tritonbench_g.cache_transform_triton.Proofs.AccessSafety
import bench.tritonbench_g.cache_transform_triton.Proofs.Grid

namespace VeriTile.Bench.Repairs.CacheTransform
open VeriTile.Triton FP.Structural
noncomputable section

/-- Existing typed gather inputs turn declared row/cache/output capacities into
bounds for every actual source access, including inactive metadata lanes. -/
theorem access_safe_of_input {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (n capacity : Nat) (rows cache : Nat → Nat)
    (input : GatherInput p s n capacity rows cache) (positive : 0<p.count)
    (bounds : RegionName → Nat) (table : n≤bounds p.rows)
    (payload : capacity*p.rowBytes≤bounds p.cache) (output : p.count≤bounds p.output) :
    TraceSafe M bounds (source p) s := by
  have activeSafe (i : TileIndex [p.block]) : safe p s i<p.count :=
    safe_bounds p.count (index p s i) positive
  have rowBound (i : TileIndex [p.block]) : safe p s i/p.rowBytes<n := by
    apply (Nat.div_lt_iff_lt_mul input.width).mpr
    rw [←input.shape]
    exact activeSafe i
  apply access_safe M p s bounds
  · intro i
    exact Nat.lt_of_lt_of_le (rowBound i) table
  · intro i _
    have rowSmall : rows (safe p s i/p.rowBytes)<18446744073709551616 :=
      lt_trans (lt_of_lt_of_le (input.rowBound _ (rowBound i)) input.capacityBound) (by decide)
    have offset : inputOffset p s i =
        rows (safe p s i/p.rowBytes)*p.rowBytes+safe p s i%p.rowBytes := by
      unfold inputOffset
      rw [input.table _ (rowBound i),Cell.read_same]
      rw [Int.emod_eq_of_lt (Int.natCast_nonneg _) (by exact_mod_cast rowSmall),Int.toNat_natCast]
    rw [offset]
    apply Nat.lt_of_lt_of_le _ payload
    exact (gather_bounds n p.rowBytes capacity (safe p s i) rows input.width
      (input.shape ▸ positive) (input.shape ▸ activeSafe i) input.rowBound).2.2
  · intro i _
    exact Nat.lt_of_lt_of_le (activeSafe i) output

end
end VeriTile.Bench.Repairs.CacheTransform
