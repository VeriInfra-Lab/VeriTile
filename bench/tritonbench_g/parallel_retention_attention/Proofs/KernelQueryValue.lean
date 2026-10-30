import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelBinding
import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelQueryWhole

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
open VeriTile.Triton FP.Structural
open Native
noncomputable section

theorem eval_outputGradient (a : Scalar α) (p : DotTrees) (s : Shape) (b : Bases)
    (m : Memory α) (tile : Tile) (valid : ProgramDomain s 64 tile) (i : TileIndex [64,s.BV]) :
    evaluate a p (FPQuery.outputGradient algebra (backwardConfig s tile) (initial s b m tile) i) =
      load a.ops s.T s.DV (currentInputs a.ops s b m).g tile.bh
        (tile.chunk*64+i.1.val) (tile.value*s.BV+i.2.1.val) := by
  change evaluate a p (matrixRead algebra (initial s b m tile) "g" (tile.bh*(s.T*s.DV))
    s.T s.DV s.DV 1 (tile.chunk*64) (tile.value*s.BV) 64 s.BV i) = _
  rw [eval_dense_read a p s b m tile "g" (s.B*s.H) s.T s.DV
    tile.bh (tile.chunk*64) (tile.value*s.BV) 64 s.BV valid.bh i]
  rfl

theorem eval_queryKey (a : Scalar α) (p : DotTrees) (s : Shape) (b : Bases)
    (m : Memory α) (tile : Tile) (valid : ProgramDomain s 64 tile) (start : Nat)
    (i : TileIndex [32,s.BK]) :
    evaluate a p (matrixRead algebra (initial s b m tile) "k" (tile.bh*(s.T*s.DK))
      s.T s.DK s.DK 1 start (tile.key*s.BK) 32 s.BK i) =
      load a.ops s.T s.DK (currentInputs a.ops s b m).k tile.bh
        (start+i.1.val) (tile.key*s.BK+i.2.1.val) := by
  rw [eval_dense_read a p s b m tile "k" (s.B*s.H) s.T s.DK
    tile.bh start (tile.key*s.BK) 32 s.BK valid.bh i]
  rfl

theorem eval_queryValue (a : Scalar α) (p : DotTrees) (s : Shape) (b : Bases)
    (m : Memory α) (tile : Tile) (valid : ProgramDomain s 64 tile) (start : Nat)
    (i : TileIndex [s.BV,32]) :
    evaluate a p (matrixRead algebra (initial s b m tile) "v" (tile.bh*(s.T*s.DV))
      s.DV s.T 1 s.DV (tile.value*s.BV) start s.BV 32 i) =
      load a.ops s.T s.DV (currentInputs a.ops s b m).v tile.bh
        (start+i.2.1.val) (tile.value*s.BV+i.1.val) := by
  rw [eval_transposed_read a p s b m tile "v" (s.B*s.H) s.T s.DV
    tile.bh start (tile.value*s.BV) s.BV 32 valid.bh i]
  rfl

#axiomsClean eval_outputGradient
#axiomsClean eval_queryKey
#axiomsClean eval_queryValue
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
