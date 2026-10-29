import bench.repairs.tritonbench_g.swiglu_fwd.Kernels

namespace VeriTile.Bench.Repairs.SwigluFwd.SourcePreservation

/-- Complete function equality retains every region, stride, extent, block
configuration and kernel header, without an input-domain premise. -/
theorem original : kernel = sourceKernel .fp32 .fp32 .fp32 Bool.false := rfl

theorem indexCommuted : indexCommutedKernel =
    sourceKernel .fp32 .fp32 .fp32 Bool.true := rfl

end VeriTile.Bench.Repairs.SwigluFwd.SourcePreservation
