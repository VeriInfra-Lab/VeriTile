import bench.tritonbench_g.parallel_retention_attention.Proofs.FPBackwardComposition

namespace VeriTile.Bench.TritonBenchG.ParallelRetentionAttention.SourceCorrect
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (DotPrimitive withDot BackwardConfig)
noncomputable section

/-- Total source execution with an independently specified complete memory,
including every unchanged cell. Numerical primitives remain explicit. -/
def ComputesMemory {α : Type} [Inhabited α] (M : Algebra α) (kernel : ComputeKernel)
    (initial expected : State α) : Prop :=
  ∃ final, FP.Structural.exec M kernel initial = some final ∧ final.mem = expected.mem

/-- Complete fp32 DQ helper: both loops, both scale sites, original casts and
masked stores. exp2/log2/dot/scalar and native integer bindings are deferred in
NUMERICAL_ASSUMPTIONS.json, not assumed to have passed experiments. -/
specification query_fp32 {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (dq : RegionName) (s : State α) :
    ComputesMemory (withDot M D) (FPQuery.source c dq) s
      (FPQuery.outputMemory M c dq s (FPQuery.values M D c s)) := by
  obtain ⟨final, he, hm, _⟩ := FPQuery.query_fp32 M D c dq s
  exact ⟨final, he, hm⟩

/-- DQ followed by DK/DV, in the original backward call order after index
specialization. Arbitrary aliases are retained: later reads see DQ's writes.
Host sums across NK/NV, inter-program scheduling, physical allocation and
the native barrier are separate obligations, not conclusions of this theorem. -/
specification backward_helpers_fp32 {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (dq : RegionName) (s : State α) :
    ComputesMemory (withDot M D) (FPBackward.source c dq) s (FPBackward.memory M D c dq s) := by
  exact FPBackward.backward_fp32 M D c dq s

#axiomsClean query_fp32
#axiomsClean backward_helpers_fp32
end
end VeriTile.Bench.TritonBenchG.ParallelRetentionAttention.SourceCorrect
