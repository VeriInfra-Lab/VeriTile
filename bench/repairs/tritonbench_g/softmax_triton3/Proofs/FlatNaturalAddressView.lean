import bench.repairs.tritonbench_g.softmax_triton3.Proofs.NaturalAddressView
namespace VeriTile.Bench.Repairs.SoftmaxTriton3.NaturalAddressView
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1200000
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
    by_cases hi : active i <;> simp only [hi,ite_true,ite_false,ih] <;> rfl

theorem flat_memory (out inp mask : RegionName) (stride n B : Nat) (hasMask : Bool)
    (hB : 0 < B) (A : FlatAlloc) (s : BlockState) :
    (exec (A.flattenKernel (source out inp mask stride n B hasMask).toAlgKernel) s).map BlockState.mem =
      (exec (A.flattenKernel (view out inp mask stride n B hasMask).toAlgKernel) s).map BlockState.mem := by
  cases hasMask <;>
    simp only [view,source,TritonBenchG.SoftmaxTriton3.softmax_kernel,SoftmaxTriton3.softmax_kernel,ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.bind,bind]
  all_goals
    simp only [FlatAlloc.flattenKernel,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,
      FlatAlloc.flattenOp.eq_def,FlatAlloc.flattenAccess,FlatAlloc.flattenMask,FlatAlloc.shiftBroadcast]
  all_goals simp [FlatAlloc.flattenKernel,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp.eq_def,
      FlatAlloc.flattenAccess,FlatAlloc.flattenMask,FlatAlloc.shiftBroadcast,view,source,TritonBenchG.SoftmaxTriton3.softmax_kernel,SoftmaxTriton3.softmax_kernel,
      ComputeKernel.surfaceBody,ComputeKernel.toAlgKernel,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
      exec,stepStmts,stepStmt,evalOp.eq_def,FloatDType.cast,
      Tile.bop,Tile.cop,Tile.ptrAdd,Tile.ptrSub,Tile.uop,
      Tile.reduceMax,Tile.reduceMaxDrop,Tile.reduceSum,Tile.reduceSumDrop,
      TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
      NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,ComparableDType.lt,
      Region.cast,hB,←Int.natCast_mul,←Int.natCast_add,-Nat.cast_mul,-Nat.cast_add,
      Nat.add_assoc,fold_mem]

  all_goals rfl

#axiomsClean flat_memory

end
end VeriTile.Bench.Repairs.SoftmaxTriton3.NaturalAddressView
