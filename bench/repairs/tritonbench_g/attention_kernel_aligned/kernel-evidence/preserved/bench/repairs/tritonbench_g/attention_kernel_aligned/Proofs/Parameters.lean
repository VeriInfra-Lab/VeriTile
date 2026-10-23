import bench.repairs.tritonbench_g.attention_kernel_aligned.Kernels
import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.Primitives
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Full
open VeriTile.Triton FP.Structural
noncomputable section
theorem block_guard (p : Parameters) (c : Nat) (hBN : 0 < p.blockN) :
    c * p.blockN < p.length ↔ c < p.blocks := by
  unfold Parameters.blocks
  rw [Nat.lt_div_iff_mul_lt hBN]
  have hs : (p.length + p.blockN - 1) - (p.blockN - 1) = p.length := by omega
  rw [hs]

#axiomsClean block_guard
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Full
