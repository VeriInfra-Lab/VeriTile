import bench.tritonbench_g.parallel_retention_attention.Proofs.FPShape
import bench.tritonbench_g.parallel_retention_attention.Proofs.FPForwardWhole
namespace VeriTile.Bench.TritonBenchG.ParallelRetentionAttention.FPCorrect
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (DotPrimitive withDot ForwardConfig BackwardConfig backwardOutputMemory)
open FPStages
noncomputable section

/-- Complete original accepted forward O source scope at fp32 storage.
The precision-selected dot/exp2/log2 backend remains an explicit binding. -/
specification forward_fp32 {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s : State α) :
    ∃ final, FP.Structural.exec (withDot M D) (forwardKernel c) s = some final ∧
      final.mem = (outputMemory M c s (forwardValues M D c s)).mem := by
  exact FPStages.forward_fp32 M D c s

/-- Complete original accepted backward DK/DV stage, including descending
and diagonal loops and the full memory frame. -/
specification backward_fp32 {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s : State α) :
    ∃ final, FP.Structural.exec (withDot M D) (backwardKernel c) s = some final ∧
      final.mem = (backwardOutputMemory M c s (backwardValues M D c s)).mem := by
  exact FPStages.backward_fp32 M D c s

#axiomsClean forward_fp32
#axiomsClean backward_fp32
end
end VeriTile.Bench.TritonBenchG.ParallelRetentionAttention.FPCorrect
