import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Correct
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.Kernels
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.Contract

/-! The F1 review finding concerns native replicas. These exact source/Real
proofs and mathematical exceptional scope were accepted in the v1 review and
are reused without changing the source, guards, input words or theorem. -/
namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Correct
abbrev checked_forward := SoftmaxFlaggemsFiniteSeed.Correct.checked_forward
abbrev checked_backward := SoftmaxFlaggemsFiniteSeed.Correct.checked_backward
abbrev saved_vjp := SoftmaxFlaggemsFiniteSeed.Correct.saved_vjp
#axiomsClean checked_forward
#axiomsClean checked_backward
#axiomsClean saved_vjp
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Correct
