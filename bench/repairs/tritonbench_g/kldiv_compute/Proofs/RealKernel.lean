import bench.repairs.tritonbench_g.kldiv_compute.Proofs.Program
import bench.repairs.tritonbench_g.kldiv_compute.Proofs.Real

namespace VeriTile.Bench.Repairs.KldivCompute.Kernel.Correct
open VeriTile.Triton FP.Structural
noncomputable section
set_option autoImplicit false

/-- Complete raw source grid. The exact memory equation gives the independent
pointwise KL result at every active output and preserves every other cell,
including zero-length grids and coincident input/output regions. -/
theorem kernel_execution (p : Parameters) (initial : State ℝ) (positive : 0 < p.block) :
    ∃ final, Program.grid Math.algebra p initial = some final ∧
      final.mem = fun r i => if r = p.output ∧ i < p.n then
        .mk .real (Math.formula ((initial.mem p.x i).read .real)
          ((initial.mem p.y i).read .real)) else initial.mem r i := by
  obtain ⟨final, executed, memory⟩ := Program.grid_complete Math.algebra p initial positive
  exact ⟨final, executed, by simpa only [Program.resultCell, Math.scalar_formula,
    VeriTile.Bench.TritonBenchG.VectorAddition.PointwisePrograms.finalMemory] using memory⟩

#axiomsClean kernel_execution
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivCompute.Kernel.Correct
