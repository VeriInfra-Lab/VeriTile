import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.ReuseEquivalence

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.FPEquiv
open VeriTile Triton FP.Structural Whole KernelNative
open scoped VeriTile.Spec
noncomputable section

def original (trees : Program → Native.Reductions) (c : Call) (m : Metadata c)
    (work : List Program) :=
  ProgramIO.Implementation.original
    (Interface.symbolicIO (KernelNative.algebras trees) c m (hostScale c.dim) work)
    (Interface.MetadataValues c m)

def blockOriginReuse (trees : Program → Native.Reductions) (c : Call) (m : Metadata c)
    (work : List Program) :=
  ProgramIO.Implementation.ofVariant
    (Interface.reuse (KernelNative.algebras trees) c m (hostScale c.dim) work)
    (Interface.MetadataValues c m)

/-- Actual complete Q/K score grids. Reuse the scalar token-block origin;
all FP32 leaves, reductions, masks, source scale and output frames stay fixed. -/
specification block_origin_reuse_equiv (trees : Program → Native.Reductions)
    (c : Call) (m : Metadata c) (valid : Valid c m)
    (work : List Program) (order : (programs c.grid).Perm work) :
    original trees c m work ≡[([] : Spec.Assumptions _)] blockOriginReuse trees c m work :=
  ProgramIO.Implementation.of_equivalentOn _
    (ReuseEquivalence.equivalent (KernelNative.algebras trees) c m (hostScale c.dim) valid work order)

#axiomsClean block_origin_reuse_equiv
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.FPEquiv
