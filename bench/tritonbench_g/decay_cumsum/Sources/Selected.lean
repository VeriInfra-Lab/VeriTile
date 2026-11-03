import bench.tritonbench_g.decay_cumsum.Sources.Body

/-! Compatibility entries for the single pasted Python source and its existing
mask-index candidates. Every invocation uses the same complete function body. -/
namespace VeriTile.Bench.Repairs.DecayCumsum.Kernels
open VeriTile.Bench.TritonBenchG.DecayCumsum.TypedSource
noncomputable section

abbrev quotedForward :=
  Bodies.forwardBody Storage.fp32

abbrev quotedForwardIndexCommuted :=
  Bodies.forwardCandidateBody Storage.fp32

abbrev quotedPrepare :=
  Bodies.prepareBody Storage.fp32 Storage.fp32

abbrev quotedPrepareIndexCommuted :=
  Bodies.prepareCandidateBody Storage.fp32 Storage.fp32

abbrev quotedBackward :=
  Bodies.backwardBody Storage.fp32 Storage.fp32

abbrev quotedBackwardIndexCommuted :=
  Bodies.backwardCandidateBody Storage.fp32 Storage.fp32

end
end VeriTile.Bench.Repairs.DecayCumsum.Kernels
