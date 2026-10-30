import bench.repairs.tritonbench_g.mean_reduction.Kernels

namespace VeriTile.Bench.Repairs.MeanReduction.SourcePreservation

/-- Equality of the complete source function, including every argument and
kernel header; no input-domain premise is needed. -/
theorem original : sourceKernel =
    VeriTile.Bench.TritonBenchG.MeanReduction.typedKernel .fp32 .fp32 Bool.false := rfl

theorem indexCommuted : indexCommutedSource =
    VeriTile.Bench.TritonBenchG.MeanReduction.typedKernel .fp32 .fp32 Bool.true := rfl

end VeriTile.Bench.Repairs.MeanReduction.SourcePreservation
