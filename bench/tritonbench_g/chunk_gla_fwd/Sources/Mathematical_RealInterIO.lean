import bench.tritonbench_g.chunk_gla_fwd.Sources.Mathematical_IO
import bench.tritonbench_g.chunk_gla_fwd.Sources.Mathematical_RealInterFlat

/-! Explicit inter-subblock IO, connected to complete finite-memory execution. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterIO
open VeriTile.Triton RealInterMath RealInterProgram RealIO
open scoped RealIO
noncomputable section
set_option maxRecDepth 12000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

theorem projection (p : Layout) (r : Regions) (qs ks gs : TypedInter.Storage)
    (out : FloatDType) (scale : ℝ) (swapped : Bool) :
    (source p r qs ks gs out scale swapped).eraseDType.toAlgorithm? =
      .ok (source p r qs ks gs out scale swapped).eraseDType.toAlgKernel := by
  simp only [source,ComputeKernel.eraseDType,RealInterSource.projects,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]

def formula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : TileIndex [p.BC,p.BC]) : ℝ :=
  expected p l a scale i.1.val i.2.1.val

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterIO
