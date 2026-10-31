import bench.tritonbench_g.chunk_gla_fwd.Sources.Mathematical_IO
import bench.tritonbench_g.chunk_gla_fwd.Sources.Mathematical_RealMergeFlat

/-! Connect the merge kernel's explicit IO to complete flat execution. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealMergeIO
open VeriTile.Triton RealMergeProgram RealIO
open scoped RealIO
noncomputable section
set_option maxRecDepth 12000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

theorem projection (p : Layout) (A A2 : RegionName) (out : FloatDType) (swapped : Bool) :
    (source p A A2 out swapped).eraseDType.toAlgorithm? =
      .ok (source p A A2 out swapped).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType,projects,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealMergeIO
