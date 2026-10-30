import bench.tritonbench_g.chunk_gated_attention.Proofs.SourcePair

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly
open VeriTile Triton FP.ProfiledRewrite SourcePair
open scoped VeriTile.Spec
noncomputable section

/-- The complete actual recurrent source and its existing sum-left candidate.
The retained reached-site domain checks the FP32 carry and dot operands before
commuting their addition. The theorem is uniform in every original plan base;
NativeCorrect retains the unrestricted native observation and full IO grid. -/
specification inner_source_equiv (c : HostFP.Inner) (base : Nat) (R : FP.ScalarArithmetic.Rules) :
    innerOriginal c base ≡[R] innerSumLeft c base :=
  SourcePair.inner_equivalent c base R

/-- The complete prefix source and existing index-first candidate, including
all batch-exhaustion branches and autotune bases. Only natural address
multiplication changes, so the numerical rule set and domain are empty. -/
specification prefix_source_equiv (c : HostFP.Cumulative) (base : Nat) :
    prefixOriginal c base ≡[indexAssumptions] prefixIndexFirst c base :=
  SourcePair.prefix_equivalent c base

#print_fp_assumptions inner_source_equiv
#print_fp_assumptions prefix_source_equiv
#axiomsClean inner_source_equiv
#axiomsClean prefix_source_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly
