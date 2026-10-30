import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.KernelBackwardStep
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.KernelLoops

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
  (foldRange BackwardAccumulator descendingCount descendingStart)
open Native
noncomputable section
set_option maxHeartbeats 10000000

theorem eval_backwardValues (a : Scalar α) (p : DotTrees) (s : Shape) (b : Bases)
    (m : Memory α) (tile : Tile) (valid : ProgramDomain s 64 tile)
    (row : Fin 64) (key : Fin s.BK) (value : Fin s.BV) :
    (evaluate a p ((FPStages.backwardValues algebra dots (backwardConfig s tile) (initial s b m tile)).key
       (row,key,PUnit.unit)),
     evaluate a p ((FPStages.backwardValues algebra dots (backwardConfig s tile) (initial s b m tile)).value
       (row,value,PUnit.unit))) =
    keyValueLane a.ops (plans s p) s (currentInputs a.ops s b m) tile row.val key.val value.val := by
  let project (z : BackwardAccumulator (Expr α) 64 s.BK s.BV) : α×α :=
    (evaluate a p (z.key (row,key,PUnit.unit)),evaluate a p (z.value (row,value,PUnit.unit)))
  have hfirst (z : BackwardAccumulator (Expr α) 64 s.BK s.BV) :
      project (foldRange (fun j=>FPStages.backwardUpdate algebra dots (backwardConfig s tile)
        (initial s b m tile) Bool.false (descendingStart (backwardConfig s tile) j))
        0 (descendingCount (backwardConfig s tile)) 1 z) =
      foldN (keyValueFuture a.ops (plans s p) s (currentInputs a.ops s b m)
        tile row.val key.val value.val) (blocks s.T 32-(tile.chunk+1)*2) (project z) := by
    rw [backward_count]
    have hp := source_range_project
      (fun j=>FPStages.backwardUpdate algebra dots (backwardConfig s tile)
        (initial s b m tile) Bool.false (descendingStart (backwardConfig s tile) j))
      (keyValueFuture a.ops (plans s p) s (currentInputs a.ops s b m) tile row.val key.val value.val)
      project (blocks s.T 32-(tile.chunk+1)*2) 1 0 (by omega)
      (fun j prev=>by
        simp only [Nat.mul_one,Nat.zero_add,backward_start]
        exact eval_backwardFuture a p s b m tile valid prev j row key value) z
    simpa only [Nat.mul_one,Nat.zero_add] using hp
  have hlast (z : BackwardAccumulator (Expr α) 64 s.BK s.BV) :
      project (foldRange (FPStages.backwardUpdate algebra dots (backwardConfig s tile)
        (initial s b m tile) Bool.true) (tile.chunk*64) (tile.chunk*64+2*32) 32 z) =
      foldN (keyValueDiagonal a.ops (plans s p) s (currentInputs a.ops s b m)
        tile row.val key.val value.val) 2 (project z) :=
    source_range_project _ _ project 2 32 (tile.chunk*64) (by omega)
      (fun j prev=>eval_backwardDiagonal a p s b m tile valid prev j row key value) z
  unfold FPStages.backwardValues keyValueLane
  change project (foldRange _ (tile.chunk*64) ((tile.chunk+1)*64) 32
    (FPStages.backwardBetweenValues algebra (backwardConfig s tile)
      (foldRange _ 0 (descendingCount (backwardConfig s tile)) 1
        ⟨fun _=>Expr.literal 0,fun _=>Expr.literal 0⟩))) = _
  rw [show (tile.chunk+1)*64=tile.chunk*64+2*32 by omega,hlast]
  dsimp only
  simp only [show a.ops.cast=id from rfl, id_eq, Prod.mk.eta]
  change foldN _ 2 _ = foldN _ 2 _
  congr 1
  have hp := hfirst ⟨fun _=>Expr.literal 0,fun _=>Expr.literal 0⟩
  have hk := congrArg Prod.fst hp
  have hv := congrArg Prod.snd hp
  apply Prod.ext <;>
    dsimp only [project,FPStages.backwardBetweenValues,backwardConfig,config,algebra,consume,evaluate,Scalar.ops]
  · apply congrArg₂ (a.binary .mul)
    · exact hk
    · apply congrArg (fun x=>a.binary .mul x (a.literal ((Real.sqrt (s.DK : ℝ))⁻¹)))
      change evaluate a p (FPStages.rowDecay algebra (FPStages.logDecay algebra (tile.bh%s.H))
        (64-row.val)) = decayNat a.ops (logDecay a.ops (tile.bh%s.H)) (64-row.val)
      rw [eval_rowDecay,eval_logDecay]
  · apply congrArg (fun x=>a.binary .mul x (a.literal ((Real.sqrt (s.DK : ℝ))⁻¹)))
    exact hv

#axiomsClean eval_backwardValues
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
