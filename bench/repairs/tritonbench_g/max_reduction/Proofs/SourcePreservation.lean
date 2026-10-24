import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.max_reduction.Kernels
import bench.repairs.tritonbench_g.max_reduction.Proofs.LegacySource
namespace VeriTile.Bench.Repairs.MaxReduction.SourcePreservation
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG

def legacyStorage : VeriTile.Bench.TritonBenchG.MaxReduction.Storage → Legacy.Storage
  | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16
  | .f8e4 => .f8e4 | .f8e5 => .f8e5 | .fp64 => .fp64

theorem stage1_preserved (storage : VeriTile.Bench.TritonBenchG.MaxReduction.Storage) (candidate : Bool) :
    VeriTile.Bench.TritonBenchG.MaxReduction.stage1 storage candidate = Legacy.stage1 (legacyStorage storage) candidate := by
  cases storage <;> cases candidate <;> rfl

theorem stage2_preserved (storage : VeriTile.Bench.TritonBenchG.MaxReduction.Storage) (candidate : Bool) :
    VeriTile.Bench.TritonBenchG.MaxReduction.stage2 storage candidate = Legacy.stage2 (legacyStorage storage) candidate := by
  cases storage <;> cases candidate <;> rfl

theorem dimension_preserved (storage : VeriTile.Bench.TritonBenchG.MaxReduction.Storage) (candidate : Bool) :
    VeriTile.Bench.TritonBenchG.MaxReduction.dimensionSource storage candidate = Legacy.dimensionSource (legacyStorage storage) candidate := by
  cases storage <;> cases candidate <;> rfl

#axiomsClean stage1_preserved
#axiomsClean stage2_preserved
#axiomsClean dimension_preserved
end VeriTile.Bench.Repairs.MaxReduction.SourcePreservation
