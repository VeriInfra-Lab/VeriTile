import bench.tritonbench_g.diag_ssm_triton.Kernels

/-! Shared entries route to the repaired Python function bodies in
Sources/Body.lean. Every cast, range, branch, mask, tuple update and store
stays in the pasted body. Dtype declarations and runtime argument bindings
remain outside the quotation. These kernels do not depend on proof modules. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.Kernels
open VeriTile.Bench.TritonBenchG.DiagSsmTriton.TypedSource

def blockSize : Nat := 128

abbrev forwardTyped :=
  Bodies.forwardBody Storage.fp32

abbrev forwardIndexCommuted :=
  Bodies.forwardCandidateBody Storage.fp32

abbrev backwardTyped :=
  Bodies.backwardBody Storage.fp32

abbrev backwardIndexCommuted :=
  Bodies.backwardCandidateBody Storage.fp32

abbrev complexForwardTyped :=
  Bodies.complexForwardBody

abbrev complexForwardIndexCommuted :=
  Bodies.complexForwardCandidateBody

abbrev complexBackwardTyped :=
  Bodies.complexBackwardBody

abbrev complexBackwardIndexCommuted :=
  Bodies.complexBackwardCandidateBody

end VeriTile.Bench.Repairs.DiagSsmTriton.Kernels
