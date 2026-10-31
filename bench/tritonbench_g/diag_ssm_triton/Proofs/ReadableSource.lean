import bench.tritonbench_g.diag_ssm_triton.Sources.Mathematical_Kernels
import bench.tritonbench_g.diag_ssm_triton.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.DiagSsmTriton.Kernels.ReadableSource
open VeriTile.Bench.TritonBenchG.DiagSsmTriton

/-- Complete function equality, including source IO, all arguments and branches. -/
theorem forward_preserved : forwardTyped=TypedSource.forward .fp32 Bool.false := rfl
theorem backward_preserved : backwardTyped=TypedSource.backward .fp32 Bool.false := rfl
theorem complex_forward_preserved : complexForwardTyped=TypedSource.complexForward Bool.false := rfl
theorem complex_backward_preserved : complexBackwardTyped=TypedSource.complexBackward Bool.false := rfl
theorem candidate_forward_preserved : forwardIndexCommuted=TypedSource.forward .fp32 Bool.true := rfl
theorem candidate_backward_preserved : backwardIndexCommuted=TypedSource.backward .fp32 Bool.true := rfl
theorem candidate_complex_forward_preserved : complexForwardIndexCommuted=TypedSource.complexForward Bool.true := rfl
theorem candidate_complex_backward_preserved : complexBackwardIndexCommuted=TypedSource.complexBackward Bool.true := rfl

#axiomsClean forward_preserved
#axiomsClean backward_preserved
#axiomsClean complex_forward_preserved
#axiomsClean complex_backward_preserved
#axiomsClean candidate_forward_preserved
#axiomsClean candidate_backward_preserved
#axiomsClean candidate_complex_forward_preserved
#axiomsClean candidate_complex_backward_preserved
end VeriTile.Bench.Repairs.DiagSsmTriton.Kernels.ReadableSource
