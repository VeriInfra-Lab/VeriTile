import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.SourceFormula

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.Reference
open VeriTile Triton FP.Structural Whole KernelNative
noncomputable section

def term (trees : Program → Native.Reductions) (c : Call) (m : Metadata c)
    (xs : Interface.Inputs Native.Word c) (b : Fin c.batches) (h : Fin c.heads)
    (t : Fin c.maxLength) : Native.Term :=
  score (algebras trees) Native.Term.leaf c m (hostScale c.dim) xs b h t

/-- All FP32 payload words are admitted. Only the existing three scalar
instruction primitives, reduction plans and contraction policy are parameters. -/
def native (I : Native.Instructions) (trees : Program → Native.Reductions)
    (fusion : Native.Schedule) (c : Call) (m : Metadata c)
    (xs : Interface.Inputs Native.Word c) : Interface.Outputs Native.Word c :=
  Interface.result fun b h t => Native.eval I fusion (scaleWord c.dim) (term trees c m xs b h t)

end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.Reference
