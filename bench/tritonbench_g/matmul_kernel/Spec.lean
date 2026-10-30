import bench.tritonbench_g.matmul_kernel.Proofs.RealSum
namespace VeriTile.Bench.Repairs.MatmulKernel
/-- Independent mathematical matrix product, on initial logical input values.
Neither tiling, padding, native instructions nor the output memory appear. -/
noncomputable def mathematical (c : Config) (m : Memory ℝ) (r n : Nat) : ℝ :=
  ∑ k∈Finset.range c.K,m .a (c.A.addr r k)*m .b (c.B.addr k n)
#specNonCircular mathematical avoiding [value,atom,block,source,run]
end VeriTile.Bench.Repairs.MatmulKernel
