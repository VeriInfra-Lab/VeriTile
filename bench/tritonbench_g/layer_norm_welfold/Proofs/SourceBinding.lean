import bench.tritonbench_g.layer_norm_welfold.Kernels
import bench.tritonbench_g.layer_norm_welfold.Proofs.LegacySource
import bench.tritonbench_g.layer_norm_welfold.Proofs.Source

/-! Checked adapter from the historical complete proof's reduction profile to
the historical source snapshot. No program/result equality is a premise. -/
namespace VeriTile.Bench.Repairs.LayerNormWelfold.SourceBinding
open VeriTile.Triton Reuse

def tree {B : Nat} : Reuse.ReductionTree B → Source.ReductionTree B
  | .leaf i => .leaf i
  | .node l r => .node (tree l) (tree r)

theorem sum_binding {B : Nat} (t : Reuse.ReductionTree B) :
    Source.sumExpr (tree t) = sumExpr t := by
  induction t with
  | leaf i => rfl
  | node l r hl hr => simp only [tree, Source.sumExpr, sumExpr, hl, hr]

theorem chunks_binding (variance : Bool) (X : RegionName) (D B start count : Nat) :
    Source.chunks variance X D B start count = chunks variance X D B start count := by
  induction count generalizing start with
  | zero => rfl
  | succ count ih =>
    simp only [Source.chunks, chunks, ih]
    rfl

theorem normalize_binding (X W Bias Y : RegionName) (D B start count : Nat) :
    Source.normalizeCode X W Bias Y D B start count = Reuse.normalizeCode X W Bias Y D B start count := by
  induction count generalizing start with
  | zero => rfl
  | succ count ih =>
    simp only [Source.normalizeCode,
      VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole.normalizeCode, ih]
    rfl

theorem row_binding (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (t : Reuse.ReductionTree B) :
    Source.rowKernel X W Bias Y Mean Rstd S D B (tree t) =
      rowKernel X W Bias Y Mean Rstd S D B t := by
  simp only [Source.rowKernel, rowKernel, Source.statisticsKernel, statisticsKernel,
    Source.sumCode, sumCode, sum_binding, chunks_binding, normalize_binding,
    Source.blockCount, VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole.blockCount]
  rfl

#axiomsClean row_binding
end VeriTile.Bench.Repairs.LayerNormWelfold.SourceBinding
