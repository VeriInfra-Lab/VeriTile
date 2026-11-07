import bench.tritonbench_g.decay_cumsum.Kernels

/-! Compatibility entries for the single pasted Python source and its existing
mask-index candidates. Every invocation uses the same complete function body. -/
namespace VeriTile.Bench.Repairs.DecayCumsum.Kernels
open VeriTile.Bench.TritonBenchG.DecayCumsum.TypedSource
noncomputable section

abbrev quotedForward :=
  forwardKernel Storage.fp32

abbrev quotedForwardIndexCommuted :=
  forwardCandidateKernel Storage.fp32

abbrev quotedPrepare :=
  prepareKernel Storage.fp32 Storage.fp32

abbrev quotedPrepareIndexCommuted :=
  prepareCandidateKernel Storage.fp32 Storage.fp32

abbrev quotedBackward :=
  backwardKernel Storage.fp32 Storage.fp32

abbrev quotedBackwardIndexCommuted :=
  backwardCandidateKernel Storage.fp32 Storage.fp32

end
end VeriTile.Bench.Repairs.DecayCumsum.Kernels
