import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelQueryValue

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
open VeriTile.Triton FP.Structural
open Native
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 26000

theorem eval_queryUpdate (a : Scalar α) (p : DotTrees) (s : Shape) (b : Bases)
    (m : Memory α) (tile : Tile) (valid : ProgramDomain s 64 tile)
    (previous : TileIndex [64,s.BK] → Expr α) (diagonal : Bool) (j : Nat)
    (row : Fin 64) (col : Fin s.BK) :
    evaluate a p (FPQuery.update algebra dots (backwardConfig s tile) (initial s b m tile)
      diagonal ((if diagonal then tile.chunk*64 else 0)+j*32) previous (row,col,PUnit.unit)) =
    queryStep a.ops (plans s p) s (currentInputs a.ops s b m) tile row.val col.val diagonal j
      (evaluate a p (previous (row,col,PUnit.unit))) := by
  cases diagonal <;>
    dsimp only [FPQuery.update,queryStep] <;>
    simp only [Bool.false_eq_true,↓reduceIte] <;>
    dsimp [backwardConfig,config,algebra,dots,consume,evaluate,dot,plans,dotRows,dotCols,Scalar.ops]
  all_goals norm_num [FPStages.blockDecay,FPStages.rowDecay,FPStages.logDecay,
    logDecay,blockDecay,decayNat,algebra,consume,evaluate,Scalar.ops,Nat.add_sub_cancel_left]
  all_goals apply dotEval_2d
  all_goals intro lane
  all_goals first
    | exact eval_queryKey a p s b m tile valid _ _
    | skip
  all_goals dsimp only [consume,evaluate,Scalar.ops]
  all_goals try apply congrArg (fun x=>a.binary .mul x (a.literal ((Real.sqrt (s.DK : ℝ))⁻¹)))
  all_goals congr 1
  all_goals first
    | apply dotEval_2d
      · intro value
        exact eval_outputGradient a p s b m tile valid _
      · intro value
        exact eval_queryValue a p s b m tile valid _ _
    | norm_num [FPStages.rowDecay,FPStages.logDecay,logDecay,decayNat,diagonalDecay,
        decayInt,algebra,consume,evaluate,Scalar.ops,Nat.add_sub_cancel_left]
  all_goals split <;> rfl

#axiomsClean eval_queryUpdate
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
