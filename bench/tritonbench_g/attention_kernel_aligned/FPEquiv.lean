import bench.tritonbench_g.attention_kernel_aligned.Proofs.Pair

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.FPEquiv
open VeriTile Triton FP.Structural Full Whole Interface
open scoped VeriTile.Spec
noncomputable section

/-- Complete original storage branch versus the early-Q scheduling candidate.
Every original native input payload, grid permutation, mask, stride, cast and
instruction graph is retained; no experimental numerical rule is used. -/
specification query_early_equiv (p : Parameters) (g : Guard p)
    (separate : Whole.Nonempty p → Separated p) (work : List Program)
    (order : (programs (grid p)).Perm work) :
    Pair.original p work ≡[[]] Pair.queryEarly p work :=
  ProgramIO.Implementation.of_equivalent (Pair.variant p work)
    (Pair.equivalent p g separate work order)

#axiomsClean query_early_equiv
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.FPEquiv

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.FPEquiv
open VeriTile Triton FP.Structural Full Whole Interface
open scoped VeriTile.Spec
noncomputable section

/-- Complete original storage branch versus the early-Q scheduling candidate.
Every original native input payload, grid permutation, mask, stride, cast and
instruction graph is retained; no experimental numerical rule is used. -/
specification query_early_equiv (p : Parameters) (g : Guard p)
    (separate : Whole.Nonempty p → Separated p) (work : List Program)
    (order : (programs (grid p)).Perm work) :
    Pair.original p work ≡[[]] Pair.queryEarly p work :=
  ProgramIO.Implementation.of_equivalent (Pair.variant p work)
    (Pair.equivalent p g separate work order)

#axiomsClean query_early_equiv
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.FPEquiv
