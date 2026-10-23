import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.IOImplements
import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Misc.NativeReference

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.IOImplements
open VeriTile Triton FP.Structural Whole IOBinding KernelNative
open VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

theorem native (I : Native.Instructions) (trees : Program → Native.Reductions)
    (fusion : Native.Schedule) (c : Call) (m : Metadata c) (v : Valid c m)
    (work : List Program) (order : (programs c.grid).Perm work) :
    (Interface.io (algebras trees) c m (hostScale c.dim) work Native.Term.leaf
      (Native.eval I fusion (scaleWord c.dim))).ImplementsOn
        (Interface.MetadataValues c m) (Reference.native I trees fusion c m) :=
  observed _ c m _ work order _ _ v _ (fun _ _ _ _ _ => rfl)

#axiomsClean native
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.IOImplements
