import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.PublicInputBinding

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.PublicIO
open VeriTile.Triton FP.Structural Kernel Kernel.Native
noncomputable section
set_option maxHeartbeats 2000000
variable {α : Type}

theorem forward_observed (a : Scalar α) (p : DotTrees) (s : Shape) (tile : Tile)
    (valid : ProgramDomain s 128 tile) (xs : ChannelValues α (forwardInputs s))
    (state : State (Expr α)) (binding : (forwardIO a p s tile).InputBinding xs state)
    (i : TileIndex [128,s.BV]) :
    evaluate a p (FPStages.forwardValues algebra dots (config s 128)
      ((forwardInvocation α s tile).initial state) i)=forwardFormula a p s tile xs (0 : Fin 1) i := by
  rcases i with ⟨row,col,⟨⟩⟩
  rw [forward_values_binding a p s tile valid xs state binding]
  change evaluate a p (FPStages.forwardValues algebra dots (config s 128)
    (initial s (packedBases s) (packedMemory a.ops.zero s (extendForward a.ops.zero s xs)) tile)
    (row,col,PUnit.unit))=forwardLane a.ops (plans s p) s (forwardLogical a.ops.zero s xs) tile row.val col.val
  rw [eval_forwardValues a p s (packedBases s) _ tile valid row col,packed_inputs,extendForward_logical]

theorem query_observed (a : Scalar α) (p : DotTrees) (s : Shape) (tile : Tile)
    (valid : ProgramDomain s 64 tile) (xs : ChannelValues α (backwardInputs s))
    (state : State (Expr α)) (binding : (backwardIO a p s tile).InputBinding xs state)
    (i : TileIndex [64,s.BK]) :
    evaluate a p (FPQuery.values algebra dots (backwardConfig s tile)
      ((backwardInvocation α s tile).initial state) i)=backwardFormula a p s tile xs (0 : Fin 3) i := by
  rcases i with ⟨row,col,⟨⟩⟩
  rw [query_values_binding a p s tile valid xs state binding]
  change evaluate a p (FPQuery.values algebra dots (backwardConfig s tile)
    (initial s (packedBases s) (packedMemory a.ops.zero s xs) tile) (row,col,PUnit.unit))=
      queryLane a.ops (plans s p) s (backwardLogical a.ops.zero s xs) tile row.val col.val
  rw [eval_queryValues a p s (packedBases s) _ tile valid row col,packed_inputs]

theorem key_observed (a : Scalar α) (p : DotTrees) (s : Shape) (tile : Tile)
    (valid : ProgramDomain s 64 tile) (valuePositive : 0<s.BV)
    (xs : ChannelValues α (backwardInputs s))
    (state : State (Expr α)) (binding : (backwardIO a p s tile).InputBinding xs state)
    (i : TileIndex [64,s.BK]) :
    evaluate a p ((FPStages.backwardValues algebra dots (backwardConfig s tile)
      ((backwardInvocation α s tile).initial state)).key i)=backwardFormula a p s tile xs (1 : Fin 3) i := by
  rcases i with ⟨row,col,⟨⟩⟩
  rw [backward_values_binding a p s tile valid xs state binding]
  have h := congrArg Prod.fst (eval_backwardValues a p s (packedBases s)
    (packedMemory a.ops.zero s xs) tile valid row col ⟨0,valuePositive⟩)
  simpa only [reference,packed_inputs,backwardFormula] using h

theorem value_observed (a : Scalar α) (p : DotTrees) (s : Shape) (tile : Tile)
    (valid : ProgramDomain s 64 tile) (keyPositive : 0<s.BK)
    (xs : ChannelValues α (backwardInputs s))
    (state : State (Expr α)) (binding : (backwardIO a p s tile).InputBinding xs state)
    (i : TileIndex [64,s.BV]) :
    evaluate a p ((FPStages.backwardValues algebra dots (backwardConfig s tile)
      ((backwardInvocation α s tile).initial state)).value i)=backwardFormula a p s tile xs (2 : Fin 3) i := by
  rcases i with ⟨row,col,⟨⟩⟩
  rw [backward_values_binding a p s tile valid xs state binding]
  have h := congrArg Prod.snd (eval_backwardValues a p s (packedBases s)
    (packedMemory a.ops.zero s xs) tile valid row ⟨0,keyPositive⟩ col)
  simpa only [reference,packed_inputs,backwardFormula] using h

#axiomsClean forward_observed
#axiomsClean query_observed
#axiomsClean key_observed
#axiomsClean value_observed
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.PublicIO
