import bench.repairs.tritonbench_g.mean_reduction.Proofs.KernelRealResult
import bench.repairs.tritonbench_g.mean_reduction.Proofs.Contract
import bench.tritonbench_g.mean_reduction.Proofs.WholeGrid

namespace VeriTile.Bench.Repairs.MeanReduction.KernelReal
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.MeanReduction Whole
noncomputable section

theorem source_is_shared (M N : Nat) : Whole.source "X" "Mean" M N = kernel M N := rfl

/-- Kernel input values after the source's existing permute/contiguous call.
 This ordinary input binding gives copied coordinates, never a mean result. -/
def InputValues (x : Tensor rank) (dims : List (Fin rank)) (nodup : dims.Nodup)
    (entry : Nat → ℝ) (s : BlockState) : Prop :=
  ∀ row : Fin (rows x.shape dims), ∀ col : Fin (columns x.shape dims),
    s.readMem "X" (row.val*columns x.shape dims+col.val) =
      compressedValue x.shape dims nodup x.stride (x.read entry) row col

theorem coordinates (x : Tensor rank) (dims : List (Fin rank)) (nodup : dims.Nodup)
    (entry : Nat → ℝ) (s : BlockState) (input : InputValues x dims nodup entry s)
    (tiles : List Nat) (schedule : tiles.Perm (List.range (Whole.gridSize (rows x.shape dims)))) :
    ∃ final, Whole.runTiles "X" "Mean" (rows x.shape dims) (columns x.shape dims) tiles s = some final ∧
      (∀ row : Fin (rows x.shape dims), final.readMem "Mean" row.val =
        mathematicalMean x.shape dims (x.read entry) ((batchLinear x.shape dims).symm row)) ∧
      (∀ (r : RegionName) o, r ≠ "Mean" ∨ rows x.shape dims ≤ o → final.mem r o = s.mem r o) := by
  obtain ⟨final, he, hv, hf⟩ := KernelRealResult.mean_grid_real (rows x.shape dims)
    (columns x.shape dims) s tiles schedule
  refine ⟨final, he, ?_, hf⟩
  intro row
  rw [hv row.val row.isLt]
  unfold rowMean
  simp only [input row]
  exact compressed_mean_eq_mathematical x.shape dims nodup x.stride (x.read entry) row

#axiomsClean source_is_shared
#axiomsClean coordinates
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MeanReduction.KernelReal
