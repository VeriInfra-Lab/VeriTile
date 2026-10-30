import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelBackwardValue

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (BackwardAccumulator matrixTranspose)
open Native
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 26000

theorem eval_backwardFuture (a : Scalar α) (p : DotTrees) (s : Shape) (b : Bases)
    (m : Memory α) (tile : Tile) (valid : ProgramDomain s 64 tile)
    (previous : BackwardAccumulator (Expr α) 64 s.BK s.BV) (j : Nat)
    (row : Fin 64) (key : Fin s.BK) (value : Fin s.BV) :
    (evaluate a p ((FPStages.backwardUpdate algebra dots (backwardConfig s tile) (initial s b m tile)
      Bool.false ((blocks s.T 32-1-j)*32) previous).key (row,key,PUnit.unit)),
     evaluate a p ((FPStages.backwardUpdate algebra dots (backwardConfig s tile) (initial s b m tile)
      Bool.false ((blocks s.T 32-1-j)*32) previous).value (row,value,PUnit.unit))) =
    keyValueFuture a.ops (plans s p) s (currentInputs a.ops s b m) tile row.val key.val value.val j
      (evaluate a p (previous.key (row,key,PUnit.unit)),
       evaluate a p (previous.value (row,value,PUnit.unit))) := by
  apply Prod.ext <;>
    dsimp [FPStages.backwardUpdate,keyValueFuture,backwardConfig,config,
      algebra,dots,consume,evaluate,dot,plans,dotRows,dotCols,Scalar.ops,matrixTranspose]
  all_goals norm_num [FPStages.blockDecay,FPStages.rowDecay,FPStages.logDecay,
    logDecay,blockDecay,decayNat,algebra,consume,evaluate,Scalar.ops]
  all_goals apply dotEval_2d
  all_goals intro lane
  all_goals dsimp only [evaluate]
  · apply dotEval_2d
    · intro feature
      exact eval_backwardValue a p s b m tile valid _
    · intro feature
      congr 1
      exact eval_backwardGradient a p s b m tile valid _ _
  · exact eval_backwardQuery a p s b m tile valid _ _
  · apply dotEval_2d
    · intro feature
      exact eval_decayedKey a p s b m tile valid _
    · intro feature
      exact eval_backwardQuery a p s b m tile valid _ _
  · congr 1
    exact eval_backwardGradient a p s b m tile valid _ _

theorem eval_backwardDiagonal (a : Scalar α) (p : DotTrees) (s : Shape) (b : Bases)
    (m : Memory α) (tile : Tile) (valid : ProgramDomain s 64 tile)
    (previous : BackwardAccumulator (Expr α) 64 s.BK s.BV) (j : Nat)
    (row : Fin 64) (key : Fin s.BK) (value : Fin s.BV) :
    (evaluate a p ((FPStages.backwardUpdate algebra dots (backwardConfig s tile) (initial s b m tile)
      Bool.true (tile.chunk*64+j*32) previous).key (row,key,PUnit.unit)),
     evaluate a p ((FPStages.backwardUpdate algebra dots (backwardConfig s tile) (initial s b m tile)
      Bool.true (tile.chunk*64+j*32) previous).value (row,value,PUnit.unit))) =
    keyValueDiagonal a.ops (plans s p) s (currentInputs a.ops s b m) tile row.val key.val value.val j
      (evaluate a p (previous.key (row,key,PUnit.unit)),
       evaluate a p (previous.value (row,value,PUnit.unit))) := by
  apply Prod.ext <;>
    dsimp [FPStages.backwardUpdate,keyValueDiagonal,backwardConfig,config,
      algebra,dots,consume,evaluate,dot,plans,dotRows,dotCols,Scalar.ops,matrixTranspose]
  all_goals norm_num [FPStages.blockDecay,FPStages.rowDecay,FPStages.logDecay,
    logDecay,blockDecay,decayNat,algebra,consume,evaluate,Scalar.ops,Nat.add_sub_cancel_left]
  all_goals apply dotEval_2d
  all_goals intro lane
  all_goals first
    | exact eval_backwardQuery a p s b m tile valid _ _
    | exact eval_backwardGradient a p s b m tile valid _ _
    | skip
  all_goals dsimp only [evaluate]
  all_goals congr 1
  all_goals first
    | apply dotEval_2d
      · intro feature
        first
          | exact eval_backwardValue a p s b m tile valid _
          | exact eval_backwardKey a p s b m tile valid _
      · intro feature
        first
          | exact eval_backwardQuery a p s b m tile valid _ _
          | exact eval_backwardGradient a p s b m tile valid _ _
    | norm_num [diagonalDecay,decayInt,algebra,consume,evaluate,Scalar.ops]
  all_goals split <;> simp_all [evaluate,sub_eq_neg_add]

#axiomsClean eval_backwardFuture
#axiomsClean eval_backwardDiagonal
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
