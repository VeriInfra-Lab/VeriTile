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

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Source
open VeriTile.Triton Full
noncomputable section
def prologue (p : Parameters) := (source p).surfaceBody.take 23
def body (p : Parameters) : List ComputeStmt :=
  match ((source p).surfaceBody.drop 23).head? with
  | some (.forRangeDyn _ _ _ _ b) => b
  | _ => []
def epilogue (p : Parameters) := (source p).surfaceBody.drop 24
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Source

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Source
open VeriTile.Triton Full
noncomputable section
def prologue (p : Parameters) := (source p).surfaceBody.take 23
def body (p : Parameters) : List ComputeStmt :=
  match ((source p).surfaceBody.drop 23).head? with
  | some (.forRangeDyn _ _ _ _ b) => b
  | _ => []
def epilogue (p : Parameters) := (source p).surfaceBody.drop 24
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Source
