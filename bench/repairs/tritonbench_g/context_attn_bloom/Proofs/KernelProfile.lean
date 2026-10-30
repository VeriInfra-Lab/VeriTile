import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.NativePlan
import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.Bytes

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.KernelNative
open VeriTile.Triton FP.Structural Native
noncomputable section

def profile (a : Host.Arguments) : Bytes.Profile where
  plan := plan a
  storedOwner := fun _ lane => lane.2.1.val

end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.KernelNative
