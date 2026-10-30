import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelBinding
import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelBackwardComposition

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
open VeriTile.Triton FP.Structural
open Native
noncomputable section

theorem eval_backwardKey (a : Scalar α) (p : DotTrees) (s : Shape) (b : Bases)
    (m : Memory α) (tile : Tile) (valid : ProgramDomain s 64 tile) (i : TileIndex [64,s.BK]) :
    evaluate a p (backwardKey algebra (backwardConfig s tile) (initial s b m tile) i) =
      load a.ops s.T s.DK (currentInputs a.ops s b m).k tile.bh
        (tile.chunk*64+i.1.val) (tile.key*s.BK+i.2.1.val) := by
  change evaluate a p (matrixRead algebra (initial s b m tile) "k" (tile.bh*(s.T*s.DK))
    s.T s.DK s.DK 1 (tile.chunk*64) (tile.key*s.BK) 64 s.BK i) = _
  rw [eval_dense_read a p s b m tile "k" (s.B*s.H) s.T s.DK
    tile.bh (tile.chunk*64) (tile.key*s.BK) 64 s.BK valid.bh i]
  rfl

theorem eval_backwardValue (a : Scalar α) (p : DotTrees) (s : Shape) (b : Bases)
    (m : Memory α) (tile : Tile) (valid : ProgramDomain s 64 tile) (i : TileIndex [64,s.BV]) :
    evaluate a p (backwardValue algebra (backwardConfig s tile) (initial s b m tile) i) =
      load a.ops s.T s.DV (currentInputs a.ops s b m).v tile.bh
        (tile.chunk*64+i.1.val) (tile.value*s.BV+i.2.1.val) := by
  change evaluate a p (matrixRead algebra (initial s b m tile) "v" (tile.bh*(s.T*s.DV))
    s.T s.DV s.DV 1 (tile.chunk*64) (tile.value*s.BV) 64 s.BV i) = _
  rw [eval_dense_read a p s b m tile "v" (s.B*s.H) s.T s.DV
    tile.bh (tile.chunk*64) (tile.value*s.BV) 64 s.BV valid.bh i]
  rfl

theorem eval_backwardQuery (a : Scalar α) (p : DotTrees) (s : Shape) (b : Bases)
    (m : Memory α) (tile : Tile) (valid : ProgramDomain s 64 tile) (start : Nat)
    (i : TileIndex [s.BK,32]) :
    evaluate a p (matrixRead algebra (initial s b m tile) "q" (tile.bh*(s.T*s.DK))
      s.DK s.T 1 s.DK (tile.key*s.BK) start s.BK 32 i) =
      load a.ops s.T s.DK (currentInputs a.ops s b m).q tile.bh
        (start+i.2.1.val) (tile.key*s.BK+i.1.val) := by
  rw [eval_transposed_read a p s b m tile "q" (s.B*s.H) s.T s.DK
    tile.bh start (tile.key*s.BK) s.BK 32 valid.bh i]
  rfl

theorem eval_backwardGradient (a : Scalar α) (p : DotTrees) (s : Shape) (b : Bases)
    (m : Memory α) (tile : Tile) (valid : ProgramDomain s 64 tile) (start : Nat)
    (i : TileIndex [s.BV,32]) :
    evaluate a p (matrixRead algebra (initial s b m tile) "g" (tile.bh*(s.T*s.DV))
      s.DV s.T 1 s.DV (tile.value*s.BV) start s.BV 32 i) =
      load a.ops s.T s.DV (currentInputs a.ops s b m).g tile.bh
        (start+i.2.1.val) (tile.value*s.BV+i.1.val) := by
  rw [eval_transposed_read a p s b m tile "g" (s.B*s.H) s.T s.DV
    tile.bh start (tile.value*s.BV) s.BV 32 valid.bh i]
  rfl

theorem eval_decayedKey (a : Scalar α) (p : DotTrees) (s : Shape) (b : Bases)
    (m : Memory α) (tile : Tile) (valid : ProgramDomain s 64 tile) (i : TileIndex [64,s.BK]) :
    evaluate a p (FPStages.decayedKey algebra (backwardConfig s tile) (initial s b m tile) i) =
      a.ops.mul (load a.ops s.T s.DK (currentInputs a.ops s b m).k tile.bh
        (tile.chunk*64+i.1.val) (tile.key*s.BK+i.2.1.val))
        (decayNat a.ops (logDecay a.ops (tile.bh%s.H)) (64-i.1.val)) := by
  change a.binary .mul
    (evaluate a p (backwardKey algebra (backwardConfig s tile) (initial s b m tile) i))
    (evaluate a p (FPStages.rowDecay algebra (FPStages.logDecay algebra (tile.bh%s.H)) (64-i.1.val))) = _
  rw [eval_backwardKey a p s b m tile valid i,eval_rowDecay,eval_logDecay]
  rfl

#axiomsClean eval_backwardKey
#axiomsClean eval_backwardValue
#axiomsClean eval_backwardQuery
#axiomsClean eval_backwardGradient
#axiomsClean eval_decayedKey
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
