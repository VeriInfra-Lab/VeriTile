import bench.tritonbench_g.mean_reduction.Kernels
import bench.tritonbench_g.mean_reduction.Proofs.Mathematical_WholeGrid

namespace VeriTile.Bench.Repairs.MeanReduction.KernelRealResult
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.MeanReduction Whole
noncomputable section

/-- Complete source-grid execution on raw kernel memory, with the independent
finite row mean and the full exterior memory frame. -/
theorem mean_grid_real (M N : Nat) (s : BlockState)
    (tiles : List Nat) (schedule : tiles.Perm (List.range (Whole.gridSize M))) :
    VeriTile.Spec.Real (∃ final,
      Whole.runTiles "X" "Mean" M N tiles s = some final ∧
      (∀ i, i < M → final.readMem "Mean" i = Whole.rowMean N (s.readMem "X") i) ∧
      (∀ (r : RegionName) o, r ≠ "Mean" ∨ M ≤ o → final.mem r o = s.mem r o)) :=
  Whole.grid_result "X" "Mean" M N (by decide) tiles schedule s

/-- All active accesses in each full DSL trace fit the input and output buffers. -/
theorem mean_kernel_trace_safe (M N : Nat) (s : BlockState) (bounds : RegionBounds)
    (input : M*N ≤ bounds "X") (output : M ≤ bounds "Mean") :
    (kernel M N).eraseDType.toAlgKernel.TraceSafeR RoundingModel.triv bounds s :=
  Whole.source_trace_safe "X" "Mean" M N s bounds input output

#axiomsClean mean_grid_real
#axiomsClean mean_kernel_trace_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MeanReduction.KernelRealResult
