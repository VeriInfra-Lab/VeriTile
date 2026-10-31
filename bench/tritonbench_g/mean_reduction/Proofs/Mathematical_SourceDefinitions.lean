import bench.tritonbench_g.mean_reduction.Sources.Mathematical_Kernels

/-! Source selectors used by the existing complete-grid proof layers. Their
original declaration names and complete values are retained outside Proofs;
loop decomposition, execution semantics and mathematical facts stay there. -/
namespace VeriTile.Bench.TritonBenchG.MeanReduction
open VeriTile.Triton
noncomputable section

namespace Whole
def source (X Mean : RegionName) (M N : Nat) : ComputeKernel :=
  typedKernel Bool.false X Mean M N 8 8
end Whole

namespace WholeFPValue
def source (X Mean : RegionName) (M N : Nat) : ComputeKernel :=
  typedKernel Bool.false X Mean M N 8 8
end WholeFPValue

end
end VeriTile.Bench.TritonBenchG.MeanReduction
