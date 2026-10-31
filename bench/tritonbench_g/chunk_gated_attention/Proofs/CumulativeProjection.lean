import bench.tritonbench_g.chunk_gated_attention.Proofs.CumulativeSource
import VeriTile.Triton.Float.Erasure
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeSource
open VeriTile.Triton
open VeriTile.Bench.Repairs.ChunkGatedAttention
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def projection (p : Parameters) (dtype out_dtype : FloatDType) (s o : RegionName) :
    {algorithm : Kernel // (kernel p dtype out_dtype s o).toAlgorithm?=Except.ok algorithm} := by
  by_cases h:p.swapped
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp only [kernel,h,NativeSource.cumIndexFirst,NativeSource.cumIndexFirstBody,NativeSource.cumOriginal,NativeSource.cumBody,ComputeKernel.withScalarArguments,
      ScalarArgument.integer,ScalarArgument.statement,ComputeKernel.surfaceBody,
      ComputeKernel.toAlgorithm?,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,ComputeDType.eraseDType,
      Except.bind,bind,List.map_cons,List.map_nil,List.cons_append,List.nil_append]
    rfl

theorem erase_connection (p : Parameters) (dtype out_dtype : FloatDType) (s o : RegionName) :
    (kernel p dtype out_dtype s o).eraseDType.toAlgKernel=
      (projection p dtype out_dtype s o).val.eraseDType := by
  simp only [ComputeKernel.eraseDType,(projection p dtype out_dtype s o).property]
  exact ComputeKernel.toAlgKernel_fromKernelBody _ _ _

#axiomsClean projection
#axiomsClean erase_connection
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeSource
