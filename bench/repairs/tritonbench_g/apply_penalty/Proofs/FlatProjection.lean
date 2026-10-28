import bench.repairs.tritonbench_g.apply_penalty.Proofs.Projection
import VeriTile.Triton.Memory.KernelMemoryTransport
namespace VeriTile.Bench.Repairs.ApplyPenalty.Projection
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ApplyPenalty
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

private theorem fold_mem {ι : Type} (items : List ι) (o : RegionName)
    (offset : ι → Nat) (value : ι → ℝ) (active : ι → Prop) [DecidablePred active]
    (u : BlockState) :
    (items.foldl (fun t i => if active i then t.writeMem o (offset i) (value i) else t) u).mem =
      items.foldl (fun mem i => if active i then
        fun r off => if r=o ∧ off=offset i then MemCell.real (value i) else mem r off
        else mem) u.mem := by
  induction items generalizing u with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    by_cases hi : active i
    · simp only [hi,ite_true,ih]
      rfl
    · simp only [hi,ite_false,ih]

/-- Actual flat executions preserve failure and every memory cell. -/
theorem flattened_algorithm (candidate : Bool) (L P F R : RegionName) (I C S : Region .nat)
    (stride skip B : Nat) (A : FlatAlloc) (s : BlockState) :
    (exec (A.flattenKernel (current candidate L P F R I C S stride skip B).toAlgKernel) s).map BlockState.mem =
    (exec (A.flattenKernel (mathematical candidate L P F R I C S stride skip B).toAlgKernel) s).map BlockState.mem := by
  cases candidate <;>
    simp only [current,mathematical,kernel,reciprocalKernel,apply_penalty,apply_penalty_reciprocal,
      ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.surfaceBody,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?,Except.bind,bind]
  all_goals
    simp only [FlatAlloc.flattenKernel,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,
      FlatAlloc.flattenOp.eq_def,FlatAlloc.flattenAccess,FlatAlloc.flattenMask,FlatAlloc.shiftBroadcast]
  all_goals
    simp [FlatAlloc.flattenKernel,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,
      FlatAlloc.flattenOp.eq_def,FlatAlloc.flattenAccess,FlatAlloc.flattenMask,FlatAlloc.shiftBroadcast,
      current,mathematical,kernel,reciprocalKernel,apply_penalty,apply_penalty_reciprocal,
      ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.bind,bind,
      exec,stepStmts,stepStmt,evalOp.eq_def,Tile.bop,Tile.cop,Tile.ptrAdd,Tile.ptrSub,Tile.uop,
      Tile.select,NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,
      ComparableDType.lt,ComparableDType.gt]
  all_goals simp only [←Int.natCast_mul,←Int.natCast_add,Nat.add_assoc,Int.toNat_natCast,Int.toNat_neg_natCast,Nat.sub_zero,Int.ofNat_lt]
  all_goals
    simp only [fold_mem]
    congr 1
    funext mem i
    by_cases ha : s.readMemValue .nat A.flat (A.base S+s.pids 0)+i.1.val <
      s.readMemValue .nat A.flat (A.base S+(s.pids 0+1))
    all_goals simp only [ha,↓reduceIte,←Int.natCast_mul,←Int.natCast_add,Nat.add_assoc,Int.toNat_natCast,Int.toNat_neg_natCast,Nat.sub_zero]



#axiomsClean flattened_algorithm
end
end VeriTile.Bench.Repairs.ApplyPenalty.Projection
