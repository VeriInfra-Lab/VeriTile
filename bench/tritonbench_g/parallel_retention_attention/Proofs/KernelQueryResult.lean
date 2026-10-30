import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelQueryStep
import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelLoops

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
open Native
noncomputable section
set_option maxHeartbeats 8000000

theorem eval_queryValues (a : Scalar α) (p : DotTrees) (s : Shape) (b : Bases)
    (m : Memory α) (tile : Tile) (valid : ProgramDomain s 64 tile)
    (row : Fin 64) (col : Fin s.BK) :
    evaluate a p (FPQuery.values algebra dots (backwardConfig s tile) (initial s b m tile)
      (row,col,PUnit.unit)) =
    queryLane a.ops (plans s p) s (currentInputs a.ops s b m) tile row.val col.val := by
  have hrange (d : Bool) (n : Nat) (z : TileIndex [64,s.BK] → Expr α) :
      evaluate a p ((foldRange (FPQuery.update algebra dots (backwardConfig s tile)
        (initial s b m tile) d) (if d then tile.chunk*64 else 0)
        ((if d then tile.chunk*64 else 0)+n*32) 32 z) (row,col,PUnit.unit)) =
      foldN (queryStep a.ops (plans s p) s (currentInputs a.ops s b m)
        tile row.val col.val d) n (evaluate a p (z (row,col,PUnit.unit))) :=
    source_range_project _ _ (fun z=>evaluate a p (z (row,col,PUnit.unit))) n 32
      (if d then tile.chunk*64 else 0) (by omega)
      (fun j z=>eval_queryUpdate a p s b m tile valid z d j row col) z
  have hfirst := hrange Bool.false (tile.chunk*2) (fun _=>Expr.literal 0)
  have bound : tile.chunk*2*32=tile.chunk*64 := by omega
  simp only [Bool.false_eq_true,↓reduceIte,Nat.zero_add,bound,evaluate] at hfirst
  have hlast := hrange Bool.true 2
  simp only [↓reduceIte] at hlast
  unfold FPQuery.values queryLane
  change evaluate a p ((foldRange _ (tile.chunk*64) ((tile.chunk+1)*64) 32
    (FPQuery.betweenValues algebra (backwardConfig s tile)
      (foldRange _ 0 (tile.chunk*64) 32 (fun _=>Expr.literal 0))))
      (row,col,PUnit.unit)) = _
  rw [show (tile.chunk+1)*64=tile.chunk*64+2*32 by omega,hlast]
  congr 1
  change a.binary .mul (evaluate a p ((foldRange _ 0 (tile.chunk*64) 32
    (fun _=>Expr.literal 0)) (row,col,PUnit.unit)))
    (a.binary .mul (evaluate a p (FPStages.rowDecay algebra
      (FPStages.logDecay algebra (tile.bh%s.H)) row.val))
      (a.literal ((Real.sqrt (s.DK : ℝ))⁻¹))) = _
  apply congrArg₂ (a.binary .mul)
  · exact hfirst
  · rw [eval_rowDecay,eval_logDecay]
    rfl

#axiomsClean eval_queryValues
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
