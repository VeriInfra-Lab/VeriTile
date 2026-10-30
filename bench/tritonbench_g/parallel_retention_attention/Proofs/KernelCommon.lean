import bench.tritonbench_g.parallel_attention.Sources.Mathematical_FPBackwardGraph
import VeriTile.Meta.StatementAudit

/-! Task-local zero-padded load expressions for the repaired retention source.
The shared DSL and interpreter are unchanged. Original unpadded source proofs
remain in the historical benchmark directory. -/
namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
  (ForwardConfig BackwardConfig queryChunk keyChunk valueChunk)
noncomputable section

def matrixRead {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (R : RegionName) (base rows cols rowStride colStride rowOffset colOffset m n : Nat) :
    TileIndex [m,n] → α := fun i =>
  let row := rowOffset+i.1.val
  let col := colOffset+i.2.1.val
  let address := base+row*rowStride+col*colStride
  M.fp32Load (if row<rows ∧ col<cols then (s.mem R address).read .real
    else M.literal (some .fp32) .real 0)

def forwardQuery {α : Type} [Inhabited α] (M : Algebra α) (c : ForwardConfig)
    (s : State α) : TileIndex [c.BTL,c.BK] → α := fun i =>
  M.cast (some .fp32) .real .real (M.binary (some .fp32) .real .mul
    (matrixRead M s c.q (s.pids 2*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
      (queryChunk c s) (keyChunk c s) c.BTL c.BK i)
    (M.literal (some .fp32) .real c.scale))

def backwardKey {α : Type} [Inhabited α] (M : Algebra α) (c : BackwardConfig) (s : State α) :=
  matrixRead M s c.k (c.i_bh*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
    (c.i_c*c.BTL) (c.i_k*c.BK) c.BTL c.BK

def backwardValue {α : Type} [Inhabited α] (M : Algebra α) (c : BackwardConfig) (s : State α) :=
  matrixRead M s c.v (c.i_bh*c.s_vo_h) c.T c.V c.s_vo_t c.s_vo_d
    (c.i_c*c.BTL) (c.i_v*c.BV) c.BTL c.BV

end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
