import bench.tritonbench_g.chunk_cumsum_vector.IO
import bench.tritonbench_g.chunk_cumsum_vector.Proofs.PublicRealIO

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.PublicRealIO
open VeriTile.Triton
open scoped BigOperators VeriTile.Triton.LogicalScheduledKernelIO
noncomputable section

theorem logical_prefix_iff (h : Call) (c : Config) (p : Geometry.Program h) :
    (prefixLogicalIO h c p ⊨ fun xs cell =>
      ∑ time : Fin h.T, if time ≤ cell.1 then xs time cell.2 else 0) ↔
    (io h c p).RegionalScheduledImplements (fun _ => [launch h p])
      (fun xs _ j => formula h (xs (0 : Fin 1)) j) := by
  change (io h c p).RegionalScheduledImplements (fun _ => [launch h p])
    (fun values _ j => ∑ time : Fin h.T,
      if time ≤ (finProdFinEquiv.symm j).1 then
        values 0 (finProdFinEquiv (time, (finProdFinEquiv.symm j).2)) else 0) ↔ _
  have target_eq : (fun (values : (io h c p).Ctx) (_ : Fin 1) (j : Fin (h.T*BS)) =>
      ∑ time : Fin h.T, if time ≤ (finProdFinEquiv.symm j).1 then
        values 0 (finProdFinEquiv (time, (finProdFinEquiv.symm j).2)) else 0) =
      (fun xs _ j => formula h (xs (0 : Fin 1)) j) := by
    funext values channel j
    simp only [formula, mathematical, Finset.sum_filter, Finset.sum_fin_eq_sum_range]
    apply Finset.sum_congr rfl
    intro k hk
    have inside := Finset.mem_range.mp hk
    simp [inside, Fin.le_iff_val_le_val, finProdFinEquiv, pair, columnIndex, Nat.add_comm, Nat.mul_comm]
  rw [target_eq]

theorem logical_implements (h : Call) (layout : Geometry.OutputLayout h) (c : Config)
    (chosen : c∈configurations) (p : Geometry.Program h) :
    prefixLogicalIO h c p ⊨ fun xs cell =>
      ∑ time : Fin h.T, if time ≤ cell.1 then xs time cell.2 else 0 :=
  (logical_prefix_iff h c p).mpr (implements h layout c chosen p)

#axiomsClean logical_prefix_iff
#axiomsClean logical_implements
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.PublicRealIO
