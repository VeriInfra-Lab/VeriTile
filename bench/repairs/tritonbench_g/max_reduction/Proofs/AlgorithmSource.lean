import bench.repairs.tritonbench_g.max_reduction.IO
import bench.tritonbench_g.max_reduction.Proofs.DimensionIO
namespace VeriTile.Bench.Repairs.MaxReduction.AlgorithmSource
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

private theorem real_scatter_mem {ι : Type} (r : RegionName) (off : ι → Nat)
    (value : ι → ℝ) (active : ι → Prop) [DecidablePred active]
    (events : List ι) (s : BlockState) :
    (events.foldl (fun acc i => if active i then acc.writeMem r (off i) (value i) else acc) s).mem =
      events.foldl (fun mem i => if active i then
        fun r' o' => if r'=r ∧ o'=off i then MemCell.real (value i) else mem r' o'
        else mem) s.mem := by
  induction events generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    rw [ih]
    split_ifs <;> rfl

private theorem int_scatter_mem {ι : Type} (r : RegionName) (off : ι → Nat)
    (value : ι → Int) (active : ι → Prop) [DecidablePred active]
    (events : List ι) (s : BlockState) :
    (events.foldl (fun acc i => if active i then acc.writeMemTyped .int r (off i) (value i) else acc) s).mem =
      events.foldl (fun mem i => if active i then
        fun r' o' => if r'=r ∧ o'=off i then MemCell.of .int (value i) else mem r' o'
        else mem) s.mem := by
  induction events generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    rw [ih]
    split_ifs <;> rfl

theorem dimension_memory (A : FlatAlloc) (inp ov oi : RegionName) (M N K BM BN : Nat)
    (hN : 0 < BN) (s : BlockState) :
    (exec (A.flattenKernel (dimension inp ov oi M N K BM BN).eraseDType.toAlgKernel) s).map BlockState.mem =
    (exec (A.flattenKernel (VeriTile.Bench.TritonBenchG.MaxReduction.mathDimension Bool.false inp ov oi M N K BM BN).toAlgKernel) s).map BlockState.mem := by
  have projects : (dimension inp ov oi M N K BM BN).toAlgorithm? =
      .ok (dimension inp ov oi M N K BM BN).toAlgKernel := rfl
  simp only [ComputeKernel.eraseDType, projects, ComputeKernel.toAlgKernel_fromKernelBody]
  simp only [dimension, VeriTile.Bench.TritonBenchG.MaxReduction.mathDimension,
    Bool.false_eq_true, ↓reduceIte, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  try simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  simp [FlatAlloc.flattenKernel, FlatAlloc.flattenStmts, FlatAlloc.flattenStmt,
    FlatAlloc.flattenOp.eq_def, FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
    FlatAlloc.shiftBroadcast, Region.cast, exec, Option.bind, stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd,
    Tile.expandDim, Tile.reduceMax, Tile.reduceMaxDrop, Tile.argMaxDrop, Tile.argBestDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex, TileShape.dropInsertedIndex,
    NumericDType.add, NumericDType.mul, ComparableDType.lt, hN]
  try simp only [← Int.natCast_add, ← Int.natCast_mul, Int.ofNat_lt, Int.toNat_natCast]
  simp only [int_scatter_mem, real_scatter_mem, BlockState.setReg_mem]
  rfl


theorem stage1_memory (A : FlatAlloc) (inp out : RegionName) (N B : Nat)
    (hB : 0 < B) (s : BlockState) :
    (exec (A.flattenKernel (stage1 inp out N B).eraseDType.toAlgKernel) s).map BlockState.mem =
    (exec (A.flattenKernel (VeriTile.Bench.TritonBenchG.MaxReduction.max_kernel_1 inp out N B).toAlgKernel) s).map BlockState.mem := by
  have projects : (stage1 inp out N B).toAlgorithm? =
      .ok (stage1 inp out N B).toAlgKernel := rfl
  simp only [ComputeKernel.eraseDType, projects, ComputeKernel.toAlgKernel_fromKernelBody]
  simp only [stage1, VeriTile.Bench.TritonBenchG.MaxReduction.max_kernel_1,
    Bool.false_eq_true, ↓reduceIte, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  try simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  simp [FlatAlloc.flattenKernel, FlatAlloc.flattenStmts, FlatAlloc.flattenStmt,
    FlatAlloc.flattenOp.eq_def, FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
    FlatAlloc.shiftBroadcast, Region.cast, exec, Option.bind, stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd,
    Tile.expandDim, Tile.reduceMax, Tile.reduceMaxDrop, Tile.argMaxDrop, Tile.argBestDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex, TileShape.dropInsertedIndex,
    NumericDType.add, NumericDType.mul, ComparableDType.lt, hB]
  try simp only [← Int.natCast_add, ← Int.natCast_mul, Int.ofNat_lt, Int.toNat_natCast]
  all_goals simp only [BlockState.writeMem, BlockState.setReg_mem]

theorem stage2_memory (A : FlatAlloc) (inp out : RegionName) (N B : Nat)
    (hB : 0 < B) (s : BlockState) :
    (exec (A.flattenKernel (stage2 inp out N B).eraseDType.toAlgKernel) s).map BlockState.mem =
    (exec (A.flattenKernel (VeriTile.Bench.TritonBenchG.MaxReduction.max_kernel_2 inp out N B).toAlgKernel) s).map BlockState.mem := by
  have projects : (stage2 inp out N B).toAlgorithm? =
      .ok (stage2 inp out N B).toAlgKernel := rfl
  simp only [ComputeKernel.eraseDType, projects, ComputeKernel.toAlgKernel_fromKernelBody]
  simp only [stage2, VeriTile.Bench.TritonBenchG.MaxReduction.max_kernel_2,
    Bool.false_eq_true, ↓reduceIte, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  try simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  simp [FlatAlloc.flattenKernel, FlatAlloc.flattenStmts, FlatAlloc.flattenStmt,
    FlatAlloc.flattenOp.eq_def, FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
    FlatAlloc.shiftBroadcast, Region.cast, exec, Option.bind, stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd,
    Tile.expandDim, Tile.reduceMax, Tile.reduceMaxDrop, Tile.argMaxDrop, Tile.argBestDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex, TileShape.dropInsertedIndex,
    NumericDType.add, NumericDType.mul, ComparableDType.lt, hB]
  try simp only [← Int.natCast_add, ← Int.natCast_mul, Int.ofNat_lt, Int.toNat_natCast]
  all_goals simp only [BlockState.writeMem, BlockState.setReg_mem]

#axiomsClean dimension_memory
#axiomsClean stage1_memory
#axiomsClean stage2_memory
end
end VeriTile.Bench.Repairs.MaxReduction.AlgorithmSource
