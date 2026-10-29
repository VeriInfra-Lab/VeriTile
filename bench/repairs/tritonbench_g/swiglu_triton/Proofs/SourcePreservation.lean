import bench.repairs.tritonbench_g.swiglu_triton.Kernels

namespace VeriTile.Bench.Repairs.SwigluTriton.SourcePreservation
open VeriTile.Triton

/-- The readable selected source retains the complete historical FP32 program,
including its input/output metadata and typed statement list. -/
theorem forward : @forwardKernel =
    @VeriTile.Bench.TritonBenchG.SwigluTriton.typedFwd .fp32 .fp32 Bool.false := rfl

theorem backward : @backwardKernel =
    @VeriTile.Bench.TritonBenchG.SwigluTriton.typedBwd .fp32 .fp32 .fp32 Bool.false := rfl

theorem forward_commuted : @forwardIndexCommuted =
    @VeriTile.Bench.TritonBenchG.SwigluTriton.typedFwd .fp32 .fp32 Bool.true := rfl

theorem backward_commuted : @backwardIndexCommuted =
    @VeriTile.Bench.TritonBenchG.SwigluTriton.typedBwd .fp32 .fp32 .fp32 Bool.true := rfl

#axiomsClean forward
#axiomsClean backward
#axiomsClean forward_commuted
#axiomsClean backward_commuted
end VeriTile.Bench.Repairs.SwigluTriton.SourcePreservation
