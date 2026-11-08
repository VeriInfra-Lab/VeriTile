import bench.tritonbench_g.diag_ssm_triton.Kernels

/-! Shared entries route to the repaired Python function bodies in
Sources/Body.lean. Every cast, range, branch, mask, tuple update and store
stays in the pasted body. Dtype declarations and runtime argument bindings
remain outside the quotation. These kernels do not depend on proof modules. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.Kernels
open VeriTile.Bench.TritonBenchG.DiagSsmTriton.TypedSource

def blockSize : Nat := 128

abbrev forwardTyped :=
  forwardKernel Storage.fp32

abbrev forwardIndexCommuted :=
  forwardCandidateKernel Storage.fp32

abbrev backwardTyped :=
  backwardKernel Storage.fp32

abbrev backwardIndexCommuted :=
  backwardCandidateKernel Storage.fp32

abbrev complexForwardTyped :=
  complexForwardKernel

abbrev complexForwardIndexCommuted :=
  complexForwardCandidateKernel

abbrev complexBackwardTyped :=
  complexBackwardKernel

abbrev complexBackwardIndexCommuted :=
  complexBackwardCandidateKernel

end VeriTile.Bench.Repairs.DiagSsmTriton.Kernels
