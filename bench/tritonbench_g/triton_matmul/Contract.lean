import bench.tritonbench_g.triton_matmul.Proofs.KernelContract

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly
/-- The selected Python input domain and direct existing-output ABI. -/
abbrev InputContract := Contract.InputContract
abbrev OutputABI := KernelContract.Output
abbrev PhysicalInputs {α : Type} := KernelContract.Inputs (α:=α)
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly
