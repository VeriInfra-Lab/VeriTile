import bench.repairs.tritonbench_g.matmul_kernel.Proofs.Scheduling

namespace VeriTile.Bench.Repairs.MatmulKernel.Public

/-- Input-only metadata contract. Ordinary tensor/device/compiler realization
is explicitly external. All values, including nonfinite native words, are allowed. -/
structure Admitted (c : Config) : Prop where
  config : c.Valid
  indices : c.M=0 ∨ c.N=0 ∨ Signed32 c
end VeriTile.Bench.Repairs.MatmulKernel.Public
