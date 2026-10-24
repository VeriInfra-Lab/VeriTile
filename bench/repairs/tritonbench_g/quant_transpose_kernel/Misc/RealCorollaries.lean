import bench.repairs.tritonbench_g.quant_transpose_kernel.Proofs.KernelModels

/-! Optional mathematical applications of the independent quantization formula.
Import this module explicitly for the preserved reciprocal corollary. -/
namespace VeriTile.Bench.Repairs.QuantTransposeKernel.Correct
open KernelModels
noncomputable section

alias normalized_nonzero := KernelModels.normalized

end
end VeriTile.Bench.Repairs.QuantTransposeKernel.Correct
