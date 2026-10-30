import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelForwardValue

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
  (queryChunk keyChunk valueChunk)
open Native
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 26000

theorem eval_forwardUpdate (a : Scalar α) (p : DotTrees) (s : Shape) (b : Bases)
    (m : Memory α) (tile : Tile) (valid : ProgramDomain s 128 tile)
    (previous : TileIndex [128,s.BV] → Expr α) (diagonal : Bool) (j : Nat)
    (row : Fin 128) (col : Fin s.BV) :
    evaluate a p (FPStages.forwardUpdate algebra dots (config s 128) (initial s b m tile)
      diagonal ((if diagonal then tile.chunk*128 else 0)+j*32) previous (row,col,PUnit.unit)) =
    forwardStep a.ops (plans s p) s (currentInputs a.ops s b m) tile row.val col.val diagonal j
      (evaluate a p (previous (row,col,PUnit.unit))) := by
  have chunks := source_chunks s 128 b m tile valid
  have head : (initial s b m tile).pids 2=tile.bh := rfl
  cases diagonal <;>
    dsimp only [FPStages.forwardUpdate,forwardStep] <;>
    simp only [Bool.false_eq_true,↓reduceIte,
      chunks.1,chunks.2.1,chunks.2.2,head,Nat.zero_add,Nat.add_sub_cancel_left] <;>
    dsimp [config,algebra,dots,consume,evaluate,dot,plans,dotRows,dotCols,Scalar.ops]
  all_goals norm_num [FPStages.blockDecay,FPStages.rowDecay,FPStages.logDecay,
    logDecay,blockDecay,decayNat,algebra,consume,evaluate,Scalar.ops]
  all_goals apply dotEval_2d
  all_goals intro lane
  all_goals first
    | exact eval_forwardValue a p s 128 b m tile valid _ _
    | skip
  all_goals dsimp only [consume,evaluate,Scalar.ops]
  all_goals congr 1
  all_goals first
    | apply dotEval_2d
      · intro key
        exact eval_forwardQuery a p s 128 b m tile valid _
      · intro key
        exact eval_forwardKey a p s 128 b m tile valid _ _
    | norm_num [FPStages.rowDecay,FPStages.logDecay,logDecay,decayNat,diagonalDecay,
        decayInt,algebra,consume,evaluate,Scalar.ops,Nat.add_sub_cancel_left]
  all_goals split <;> rfl

#axiomsClean eval_forwardUpdate
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
