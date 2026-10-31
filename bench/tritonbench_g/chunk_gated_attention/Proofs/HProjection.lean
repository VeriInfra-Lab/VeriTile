import bench.tritonbench_g.chunk_gated_attention.Sources.Kernels
import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealHProgram
import VeriTile.Triton.Float.Erasure
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HSource
open VeriTile.Triton
open VeriTile.Bench.Repairs.ChunkGatedAttention
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

structure Parameters extends RealHMath.Layout where
  base : Nat
  gateK : Bool
  hasSeed : Bool
  storeFinal : Bool
  swapped : Bool := Bool.false

def kernel (p : Parameters) (r : RealHProgram.Regions)
    (dot gate history seed final : FloatDType) : ComputeKernel :=
  (if p.swapped then NativeSource.hSumLeftTyped else NativeSource.hOriginalTyped)
    dot gate history seed final r.key r.value r.gate r.history r.seed r.final
    p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature
    p.historyHead p.historyKey p.historyValue p.T p.K p.V p.BT p.BK p.BV p.NT p.base
    p.gateK p.hasSeed p.storeFinal

def projection (p : Parameters) (r : RealHProgram.Regions)
    (dot gate history seed final : FloatDType) :
    {algorithm : Kernel // (kernel p r dot gate history seed final).toAlgorithm?=Except.ok algorithm} := by
  by_cases h:p.swapped
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp only [kernel,h,NativeSource.hOriginalTyped,NativeSource.hBody,NativeSource.hSumLeftTyped,
      NativeSource.hSumLeftBody,ComputeKernel.withScalarArguments,ScalarArgument.integer,
      ScalarArgument.statement,ComputeKernel.surfaceBody,ComputeKernel.toAlgorithm?,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?,ComputeDType.eraseDType,Except.bind,bind,List.map_cons,List.map_nil,
      List.cons_append,List.nil_append]
    rfl

theorem erase_connection (p : Parameters) (r : RealHProgram.Regions)
    (dot gate history seed final : FloatDType) :
    (kernel p r dot gate history seed final).eraseDType.toAlgKernel=
      (projection p r dot gate history seed final).val.eraseDType := by
  simp only [ComputeKernel.eraseDType,(projection p r dot gate history seed final).property]
  exact ComputeKernel.toAlgKernel_fromKernelBody _ _ _

#axiomsClean projection
#axiomsClean erase_connection
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.HSource
