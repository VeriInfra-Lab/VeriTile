import bench.tritonbench_g.chunk_gla_fwd.Sources.Mathematical_IO
import bench.tritonbench_g.chunk_gla_fwd.Sources.Mathematical_RealIntraFlat

/-! Explicit streamed-column IO, connected to the full source and tagged frame. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraIO
open VeriTile.Triton RealIntraMath RealIntraProgram RealIO
open scoped RealIO
noncomputable section
set_option maxRecDepth 12000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

theorem projection (p : Layout) (r : Regions) (qs ks gs : TypedIntra.Storage)
    (scale : ℝ) (swapped : Bool) :
    (source p r qs ks gs scale swapped).eraseDType.toAlgorithm? = .ok (source p r qs ks gs scale swapped).eraseDType.toAlgKernel := by
  simp only [source,ComputeKernel.eraseDType,RealIntraSource.projects,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]

def formula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : Nat × Fin p.BC) : ℝ :=
  expected p l a scale i.2.val i.1

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIntraIO
