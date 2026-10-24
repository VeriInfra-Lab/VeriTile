import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.IOInputValues
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelNativeForward

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PublicIO
open VeriTile Triton FP.Structural KernelPacket
open KernelNative (sharedOps)
noncomputable section
variable {α : Type}
set_option maxHeartbeats 1000000

/-- Only the input record type differs between the two already-identical
forward sources; scalar operations and tensor dimensions are preserved. -/
def sharedInputs (xs : Whole001.Inputs α) : VeriTile.Bench.Repairs.ChunkRetention.Whole001.Inputs α :=
  ⟨xs.q,xs.k,xs.v,xs.seed,xs.incoming,xs.finalIncoming⟩

theorem shared_block_chain (ops : Ops α) (width : Nat) (l r : Nat→α) (n : Nat) (seed : α) :
    VeriTile.Bench.Repairs.ChunkRetention.Whole001.blockChain (sharedOps ops) width l r n seed=
      blockChain ops width l r n seed := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [VeriTile.Bench.Repairs.ChunkRetention.Whole001.blockChain,blockChain,
      KernelNative.shared_chain,ih]

theorem shared_history (ops : Ops α) (c : Shape) (xs : Whole001.Inputs α) (h n : Nat) :
    VeriTile.Bench.Repairs.ChunkRetention.Whole001.Reference.history (sharedOps ops) (KernelOutput.shape c)
      (sharedInputs xs) h n=Reference.history ops c xs h n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    funext e j
    simp only [VeriTile.Bench.Repairs.ChunkRetention.Whole001.Reference.history,Reference.history,
      VeriTile.Bench.Repairs.ChunkRetention.Whole001.Body.nextH,Body.nextH,KernelNative.shared_chain,ih]
    rfl

theorem shared_output (ops : Ops α) (c : Shape) (xs : Whole001.Inputs α) (scale : α)
    (h chunk : Nat) (history : Tile α) :
    VeriTile.Bench.Repairs.ChunkRetention.Whole001.Body.output (sharedOps ops) (KernelOutput.shape c)
      (sharedInputs xs) scale h chunk history=Body.output ops c xs scale h chunk history := by
  funext e j
  simp only [VeriTile.Bench.Repairs.ChunkRetention.Whole001.Body.output,Body.output,
    VeriTile.Bench.Repairs.ChunkRetention.Whole001.Body.score,Body.score,
    VeriTile.Bench.Repairs.ChunkRetention.Whole001.Body.outputCarry,Body.outputCarry,
    KernelNative.shared_chain,shared_block_chain]
  rfl

#axiomsClean shared_history
#axiomsClean shared_output
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PublicIO
