import bench.tritonbench_g.layer_norm_welfold.Kernels
import bench.tritonbench_g.layer_norm_welfold.Proofs.Source

/-! The proof's ordered reduction tree is bound to the actual source profile.
No executable source snapshot is maintained by this adapter. -/
namespace VeriTile.Bench.Repairs.LayerNormWelfold.SourceBinding
open VeriTile.Triton Reuse

def tree {B : Nat} : Reuse.ReductionTree B → SourceCode.ReductionTree B
  | .leaf i => .leaf i
  | .node l r => .node (tree l) (tree r)
end VeriTile.Bench.Repairs.LayerNormWelfold.SourceBinding
