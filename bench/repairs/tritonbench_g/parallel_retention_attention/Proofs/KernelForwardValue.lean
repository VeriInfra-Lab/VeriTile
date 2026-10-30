import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.KernelBinding
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.KernelForwardWhole

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
  (ForwardConfig queryChunk keyChunk valueChunk)
open Native
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000

theorem eval_forwardQuery (a : Scalar α) (p : DotTrees) (s : Shape) (L : Nat) (b : Bases)
    (m : Memory α) (tile : Tile) (valid : ProgramDomain s L tile) (i : TileIndex [L,s.BK]) :
    evaluate a p (forwardQuery algebra (config s L) (initial s b m tile) i) =
      a.ops.mul (load a.ops s.T s.DK (currentInputs a.ops s b m).q tile.bh
        (tile.chunk*L+i.1.val) (tile.key*s.BK+i.2.1.val)) (a.ops.scale s.DK) := by
  have chunks := source_chunks s L b m tile valid
  change a.binary .mul
    (evaluate a p (matrixRead algebra (initial s b m tile) "q" (tile.bh*(s.T*s.DK))
      s.T s.DK s.DK 1 (queryChunk (config s L) (initial s b m tile))
      (keyChunk (config s L) (initial s b m tile)) L s.BK i))
    (a.literal ((Real.sqrt (s.DK : ℝ))⁻¹)) = _
  rw [chunks.1,chunks.2.1,eval_dense_read a p s b m tile "q" (s.B*s.H) s.T s.DK
    tile.bh (tile.chunk*L) (tile.key*s.BK) L s.BK valid.bh i]
  rfl

theorem eval_forwardKey (a : Scalar α) (p : DotTrees) (s : Shape) (L : Nat) (b : Bases)
    (m : Memory α) (tile : Tile) (valid : ProgramDomain s L tile) (start : Nat)
    (i : TileIndex [s.BK,32]) :
    evaluate a p (matrixRead algebra (initial s b m tile) "k" (tile.bh*(s.T*s.DK))
      s.DK s.T 1 s.DK (tile.key*s.BK) start s.BK 32 i) =
      load a.ops s.T s.DK (currentInputs a.ops s b m).k tile.bh
        (start+i.2.1.val) (tile.key*s.BK+i.1.val) := by
  rw [eval_transposed_read a p s b m tile "k" (s.B*s.H) s.T s.DK
    tile.bh start (tile.key*s.BK) s.BK 32 valid.bh i]
  rfl

theorem eval_forwardValue (a : Scalar α) (p : DotTrees) (s : Shape) (L : Nat) (b : Bases)
    (m : Memory α) (tile : Tile) (valid : ProgramDomain s L tile) (start : Nat)
    (i : TileIndex [32,s.BV]) :
    evaluate a p (matrixRead algebra (initial s b m tile) "v" (tile.bh*(s.T*s.DV))
      s.T s.DV s.DV 1 start (tile.value*s.BV) 32 s.BV i) =
      load a.ops s.T s.DV (currentInputs a.ops s b m).v tile.bh
        (start+i.1.val) (tile.value*s.BV+i.2.1.val) := by
  rw [eval_dense_read a p s b m tile "v" (s.B*s.H) s.T s.DV
    tile.bh start (tile.value*s.BV) 32 s.BV valid.bh i]
  rfl

#axiomsClean eval_forwardQuery
#axiomsClean eval_forwardKey
#axiomsClean eval_forwardValue
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
