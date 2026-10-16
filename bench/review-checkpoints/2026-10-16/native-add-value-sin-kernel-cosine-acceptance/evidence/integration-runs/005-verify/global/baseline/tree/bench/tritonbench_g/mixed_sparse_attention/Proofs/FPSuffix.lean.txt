import bench.tritonbench_g.mixed_sparse_attention.Proofs.FPPrefix
import bench.tritonbench_g.mixed_sparse_attention.Proofs.FPMemory
namespace VeriTile.Bench.TritonBenchG.MixedSparseAttention.FPStages
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false


def suffix (c : Config) : List ComputeStmt := (activeBody c).drop 24


theorem suffix_run {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (c : Config) (s t : State α) (a : Accumulator α c.BM c.BD) (h : Ready M c s t a) :
    ∃ final, run (backend M D) (suffix c) t = some final ∧
      final.mem = (outputMemory M c s a).mem := by
  simp [suffix,activeBody,parsedActiveBody,parsedKernel,source,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
    numeric,bop,natLt,numericLt,backend,Region.cast,ofFloat,toFloat,store,
    h.output,h.normalizer,h.outptr,h.mask,outputPointer,outputMemory]
  apply maskedMemory_congr
  exact h.mem

#axiomsClean suffix_run
end
end VeriTile.Bench.TritonBenchG.MixedSparseAttention.FPStages
