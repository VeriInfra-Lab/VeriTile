import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.IO

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Target
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8
noncomputable section

/-- Independent dense matrix product and optional slope-1/100 Leaky ReLU.
Real semantics erase the output rounding without claiming native exactness. -/
def mathematical (g : KernelBuffers.Geometry) (A B : Nat → Nat → ℝ) (i j : Nat) : ℝ :=
  Real.activate g.enabled (∑ k∈Finset.range g.K, A i k*B k j)

end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Target
