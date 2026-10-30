import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.KernelForwardStep
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.KernelLoops

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange queryChunk)
open Native
noncomputable section
set_option maxHeartbeats 8000000

theorem eval_forwardValues (a : Scalar α) (p : DotTrees) (s : Shape) (b : Bases)
    (m : Memory α) (tile : Tile) (valid : ProgramDomain s 128 tile)
    (row : Fin 128) (col : Fin s.BV) :
    evaluate a p (FPStages.forwardValues algebra dots (config s 128) (initial s b m tile)
      (row,col,PUnit.unit)) =
    forwardLane a.ops (plans s p) s (currentInputs a.ops s b m) tile row.val col.val := by
  have hrange (d : Bool) (n : Nat) (z : TileIndex [128,s.BV] → Expr α) :
      evaluate a p ((foldRange (FPStages.forwardUpdate algebra dots (config s 128)
        (initial s b m tile) d) (if d then tile.chunk*128 else 0)
        ((if d then tile.chunk*128 else 0)+n*32) 32 z) (row,col,PUnit.unit)) =
      foldN (forwardStep a.ops (plans s p) s (currentInputs a.ops s b m)
        tile row.val col.val d) n (evaluate a p (z (row,col,PUnit.unit))) :=
    source_range_project _ _ (fun z=>evaluate a p (z (row,col,PUnit.unit))) n 32
      (if d then tile.chunk*128 else 0) (by omega)
      (fun j z=>eval_forwardUpdate a p s b m tile valid z d j row col) z
  have hfirst := hrange Bool.false (tile.chunk*4) (fun _=>Expr.literal 0)
  have bound : tile.chunk*4*32=tile.chunk*128 := by omega
  simp only [Bool.false_eq_true,↓reduceIte,Nat.zero_add,bound,evaluate] at hfirst
  have hlast := hrange Bool.true 4
  simp only [↓reduceIte] at hlast
  have chunks := source_chunks s 128 b m tile valid
  unfold FPStages.forwardValues forwardLane
  rw [chunks.1]
  change evaluate a p ((foldRange _ (tile.chunk*128) ((tile.chunk+1)*128) 32
    (FPStages.betweenValues algebra (config s 128) (initial s b m tile)
      (foldRange _ 0 (tile.chunk*128) 32 (fun _=>Expr.literal 0))))
      (row,col,PUnit.unit)) = _
  rw [show (tile.chunk+1)*128=tile.chunk*128+4*32 by omega,hlast]
  congr 1
  change a.binary .mul (evaluate a p ((foldRange _ 0 (tile.chunk*128) 32
    (fun _=>Expr.literal 0)) (row,col,PUnit.unit)))
    (evaluate a p (FPStages.rowDecay algebra (FPStages.logDecay algebra (tile.bh%s.H)) row.val)) = _
  apply congrArg₂ (a.binary .mul)
  · exact hfirst
  · rw [eval_rowDecay,eval_logDecay]

#axiomsClean eval_forwardValues
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
