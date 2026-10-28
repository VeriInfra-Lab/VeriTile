import bench.repairs.tritonbench_g.triton_softmax.Proofs.RowIO
namespace VeriTile.Bench.Repairs.TritonSoftmax.RowIO
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
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

/-- The actual pasted source and the internal natural-address proof view
have identical complete flat memory results, including failure. -/
theorem flat_memory (out inp : RegionName) (ib ob ir ic orow oc n B : Nat)
    (hB : 0 < B) (A : FlatAlloc) (s : BlockState) :
    (exec (A.flattenKernel (softmax_kernel out inp ib ob ir ic orow oc n B).toAlgKernel) s).map BlockState.mem =
      (exec (A.flattenKernel (SoftmaxTriton1.NaturalAddressView.view Bool.false out inp ib ob ir ic orow oc n B).toAlgKernel) s).map BlockState.mem := by
  simp only [SoftmaxTriton1.NaturalAddressView.view,SoftmaxTriton1.NaturalAddressView.softmax_kernel,
    softmax_kernel,ComputeKernel.surfaceBody,ComputeKernel.toAlgKernel,
    ComputeKernel.toAlgorithm?,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.bind,bind,ite_false]
  simp only [FlatAlloc.flattenKernel,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,
    FlatAlloc.flattenOp.eq_def,FlatAlloc.flattenAccess,FlatAlloc.flattenMask,FlatAlloc.shiftBroadcast]
  simp [FlatAlloc.flattenKernel,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp.eq_def,
    FlatAlloc.flattenAccess,FlatAlloc.flattenMask,FlatAlloc.shiftBroadcast,exec,stepStmts,stepStmt,evalOp.eq_def,
    Tile.bop,Tile.cop,Tile.ptrAdd,Tile.ptrSub,Tile.uop,
    Tile.reduceMax,Tile.reduceMaxDrop,Tile.reduceSum,Tile.reduceSumDrop,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,ComparableDType.lt,
    Region.cast,hB,←Int.natCast_mul,←Int.natCast_add,-Nat.cast_mul,-Nat.cast_add,
    Nat.add_assoc,fold_mem]
  rfl

/-- The same connection for the regional proof of outputs and frame. -/
theorem regional_memory (out inp : RegionName) (ib ob ir ic orow oc n B : Nat)
    (hB : 0 < B) (s : BlockState) :
    (exec (softmax_kernel out inp ib ob ir ic orow oc n B).toAlgKernel s).map BlockState.mem =
      (exec (SoftmaxTriton1.NaturalAddressView.view Bool.false out inp ib ob ir ic orow oc n B).toAlgKernel s).map BlockState.mem := by
  obtain ⟨t,u,ht,hu,hm⟩ := memory out inp ib ob ir ic orow oc n B hB s
  have same : (exec (softmax_kernel out inp ib ob ir ic orow oc n B).toAlgKernel s).map BlockState.mem =
      (exec (SoftmaxTriton1.softmax_kernel out inp ib ob ir ic orow oc n B).toAlgKernel s).map BlockState.mem := by
    simp only [ht,hu,Option.map_some,hm]
  exact same.trans (SoftmaxTriton1.NaturalAddressView.regional_memory Bool.false out inp ib ob ir ic orow oc n B hB s)

#axiomsClean flat_memory
#axiomsClean regional_memory
end
end VeriTile.Bench.Repairs.TritonSoftmax.RowIO
