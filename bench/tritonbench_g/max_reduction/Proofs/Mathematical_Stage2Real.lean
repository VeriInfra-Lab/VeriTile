import bench.tritonbench_g.max_reduction.Proofs.Mathematical_Math
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.MaxReduction
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedTileKernelIO₁
set_option maxHeartbeats 2000000

/-- The second-stage source terminates whenever its reduction tile is nonempty. -/
theorem stage2_terminates (mid out : RegionName) (M B : Nat) (hB : 0 < B)
    (s : BlockState) : ∃ t, exec (max_kernel_2 mid out M B) s = some t := by
  simp [exec, max_kernel_2, stage2, VeriTile.Bench.Repairs.MaxReduction.stage1, VeriTile.Bench.Repairs.MaxReduction.stage1Commuted, VeriTile.Bench.Repairs.MaxReduction.stage2, VeriTile.Bench.Repairs.MaxReduction.stage2Complement, VeriTile.Bench.Repairs.MaxReduction.dimension, VeriTile.Bench.Repairs.MaxReduction.dimensionCommuted,  ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.surfaceBody, ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind, ComputeDType.eraseDType, ComputeKernel.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType, stepStmts, stepStmt, evalOp.eq_def, Tile.cop, Tile.ptrAdd,
    Tile.reduceMax, Tile.reduceMaxDrop, TileShape.axisDim, TileShape.eraseAxis,
    ComparableDType.lt, hB]

/-- The scalar output store changes only output offset zero. -/
theorem stage2_frame (mid out : RegionName) (M B : Nat) (hB : 0 < B)
    (s t : BlockState) (he : exec (max_kernel_2 mid out M B) s = some t)
    (r : RegionName) (o : Nat) (hm : r ≠ out ∨ o ≠ 0) : t.mem r o = s.mem r o := by
  simp [exec, max_kernel_2, stage2, VeriTile.Bench.Repairs.MaxReduction.stage1, VeriTile.Bench.Repairs.MaxReduction.stage1Commuted, VeriTile.Bench.Repairs.MaxReduction.stage2, VeriTile.Bench.Repairs.MaxReduction.stage2Complement, VeriTile.Bench.Repairs.MaxReduction.dimension, VeriTile.Bench.Repairs.MaxReduction.dimensionCommuted,  ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.surfaceBody, ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind, ComputeDType.eraseDType, ComputeKernel.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType, stepStmts, stepStmt, evalOp.eq_def, Tile.cop, Tile.ptrAdd,
    Tile.reduceMax, Tile.reduceMaxDrop, TileShape.axisDim, TileShape.eraseAxis,
    ComparableDType.lt, hB] at he
  subst he
  rw [BlockState.writeMem_mem, if_neg ?_]
  · simp
  · rintro ⟨hr, ho⟩
    rcases hm with h | h
    · exact h hr
    · exact h ho

theorem stage2_flattenOk (mid out : RegionName) (M B : Nat) :
    (max_kernel_2 mid out M B).toAlgKernel.FlattenOk := by
  unfold Kernel.FlattenOk
  simp [max_kernel_2, stage2, VeriTile.Bench.Repairs.MaxReduction.stage1, VeriTile.Bench.Repairs.MaxReduction.stage1Commuted, VeriTile.Bench.Repairs.MaxReduction.stage2, VeriTile.Bench.Repairs.MaxReduction.stage2Complement, VeriTile.Bench.Repairs.MaxReduction.dimension, VeriTile.Bench.Repairs.MaxReduction.dimensionCommuted,  ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.surfaceBody, ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind, ComputeDType.eraseDType, ComputeKernel.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType, ComputeKernel.toAlgKernel, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk]
  simp only [Op.FlattenOk.eq_def]

theorem stage2_traceSafe (mid out : RegionName) (M B : Nat) (hB : 0 < B)
    (bounds : RegionBounds) (s : BlockState)
    (hi : ∀ i : Fin B, i.val < M → i.val < bounds mid) (ho : 0 < bounds out) :
    (max_kernel_2 mid out M B).toAlgKernel.TraceSafe bounds s := by
  simp [Kernel.TraceSafe, max_kernel_2, stage2, VeriTile.Bench.Repairs.MaxReduction.stage1, VeriTile.Bench.Repairs.MaxReduction.stage1Commuted, VeriTile.Bench.Repairs.MaxReduction.stage2, VeriTile.Bench.Repairs.MaxReduction.stage2Complement, VeriTile.Bench.Repairs.MaxReduction.dimension, VeriTile.Bench.Repairs.MaxReduction.dimensionCommuted,  ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.surfaceBody, ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind, ComputeDType.eraseDType, ComputeKernel.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType, Stmt.TraceSafeList, Stmt.TraceSafe,
    Op.SafeAt, MaskOpt.SafeAt, MaskOpt.Active, MemAccess.SafeAt,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    Op.PointerAddressesSafeOn, Op.MemorySafe, stepStmt, evalOp.eq_def,
    Tile.cop, Tile.ptrAdd, Tile.reduceMax, Tile.reduceMaxDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
    ComparableDType.lt, hB]
  refine ⟨hi, ?_, ho⟩
  simp only [Op.SafeAt.eq_def]

theorem stage2_formula (mid : RegionName) (M B : Nat) (hB : 0 < B)
    (s : BlockState) (xs : TileIndex [B] → ℝ)
    (hx : ∀ i : Fin B, i.val < M → s.readMem mid i.val = xs (i, PUnit.unit)) :
    maxKernel2Spec s mid M B = blockMaximum M B 0 xs := by
  rw [blockMaximum_eq_tile M B 0 hB]
  unfold maxKernel2Spec maxTileSpecOf
  congr 2
  apply congrArg Tile.mk
  funext i
  simp only [Nat.zero_mul, Nat.zero_add]
  by_cases hi : i.1.val < M
  · simp [hi, hx i.1 hi]
  · simp [hi]

/-- The second-stage IO uses the entire masked intermediate buffer and one
unmasked scalar output; it never receives a precomputed maximum as input. -/
noncomputable def stage2IO (mid out : RegionName) (M B : Nat) : MaskedTileKernelIO₁ where
  kernel := max_kernel_2 mid out M B
  projection := by simp only [max_kernel_2, ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]
  inp := mid
  out := out
  shape := [B]
  read := fun _ i => i.1.val
  write := fun _ _ => 0
  mask := fun _ i => i.1.val < M
  writeMask := fun _ i => i.1.val = 0

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MaxReduction
