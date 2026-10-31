import bench.tritonbench_g.ksoftmax_triton.Proofs.Mathematical_ForwardConstructors
import bench.tritonbench_g.ksoftmax_triton.Proofs.Mathematical_BackwardFP

/-! Structural equivalence of complete forward original/candidate programs.
BackwardFP supplies only the already checked natural multiplication and
pointer-add congruence rules. Every floating operation and control path is
the shared source suffix; no floating arithmetic atom is assumed.

This is a draft for root validation against the staged provenance frontend.
It makes no all-profile elaboration, successful-run, or native IEEE claim.
-/
namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.Floating
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.Floating
