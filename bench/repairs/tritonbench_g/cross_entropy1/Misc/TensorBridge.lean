import bench.repairs.tritonbench_g.cross_entropy1.Correct
import bench.repairs.tritonbench_g.cross_entropy1.Misc.NativeCorrect
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.KernelTensor

/-! Retained full actual-Tensor applications. The supporting proofs use the
same typed raw source/output/frame lemmas as the primary interfaces and derive
logical input, label, scalar ABI and destination correspondence explicitly. -/
namespace VeriTile.Bench.Repairs.CrossEntropy1.TensorBridge.Real
export KernelTensor (forward backward)
#axiomsClean forward
#axiomsClean backward
end VeriTile.Bench.Repairs.CrossEntropy1.TensorBridge.Real

namespace VeriTile.Bench.Repairs.CrossEntropy1.TensorBridge.Native
export KernelProfile (forward backward)
#axiomsClean forward
#axiomsClean backward
end VeriTile.Bench.Repairs.CrossEntropy1.TensorBridge.Native
