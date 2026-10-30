import bench.tritonbench_g.chunk_cumsum_kernel.Proofs.PublicReal
import bench.tritonbench_g.chunk_cumsum_kernel.IO

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.PublicReal
open VeriTile.Triton
open scoped BigOperators VeriTile.Triton.LogicalIO
noncomputable section

theorem logical_prefix_iff (input output : RegionName) (T : Nat) (cfg : Config) :
    (prefixLogicalIO input output T cfg ⊨ fun xs time =>
      ∑ k : Fin T, if k ≤ time then xs k else 0) ↔
    (io input output T cfg).RegionalImplements (fun _ _ vals _ => formula (vals 0)) := by
  change (io input output T cfg).RegionalImplements
    (fun _ _ vals _ time => ∑ k : Fin T, if k ≤ time then vals 0 k else 0) ↔ _
  have target_eq : (fun (vals : (io input output T cfg).Ctx) (_ : Fin 1) (time : Fin T) =>
      ∑ k : Fin T, if k ≤ time then vals 0 k else 0) =
      (fun vals _ => formula (vals 0)) := by
    funext vals channel time
    simp only [formula, mathematical, Finset.sum_filter, Finset.sum_fin_eq_sum_range]
    apply Finset.sum_congr rfl
    intro k hk
    have inside := Finset.mem_range.mp hk
    simp [inside, Fin.le_iff_val_le_val]
  rw [target_eq]

theorem logical_implements (input output : RegionName) (T : Nat) (cfg : Config)
    (separate : output ≠ input) :
    prefixLogicalIO input output T cfg ⊨ fun xs time =>
      ∑ k : Fin T, if k ≤ time then xs k else 0 :=
  (logical_prefix_iff input output T cfg).mpr (implements input output T cfg separate)

#axiomsClean logical_prefix_iff
#axiomsClean logical_implements
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.PublicReal
