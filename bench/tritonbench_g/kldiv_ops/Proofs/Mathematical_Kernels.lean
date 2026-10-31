import bench.tritonbench_g.kldiv_ops.Kernels
/-! Compatibility bindings to the shared repaired backward dtype family. -/
namespace VeriTile.Bench.TritonBenchG.KldivOps.TypedSource
open VeriTile.Triton
abbrev Backward := RegionName → RegionName → Nat → Nat → Nat → Nat → ComputeKernel
def backwardDefaultOriginal (target output : Storage) : Backward :=
  VeriTile.Bench.Repairs.KldivOps.BackwardSource.defaultTarget target output Bool.false
def backwardLogOriginal (target : MathStorage) (output : Storage) : Backward :=
  VeriTile.Bench.Repairs.KldivOps.BackwardSource.logarithmic target output Bool.false
abbrev backwardDefault := VeriTile.Bench.Repairs.KldivOps.BackwardSource.defaultTarget
abbrev backwardLog := VeriTile.Bench.Repairs.KldivOps.BackwardSource.logarithmic
end VeriTile.Bench.TritonBenchG.KldivOps.TypedSource
