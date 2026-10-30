import VeriTile.Triton.Memory.Flatten
import bench.repairs.tritonbench_g.dropout_triton.Proofs.LegacyPastedSurface
import bench.tritonbench_g.dropout_triton.Kernels
import bench.tritonbench_g.vector_addition.Proofs.TypedFloat
import VeriTile.Meta.StatementAudit

/-! Complete source connections. The historical quote has a redundant store
cast absent from Python. Same-precision cast identity is a DEFINED part of the
already-used `TypedFloat.castBackend`, not a law postulated for arbitrary M.
The actual original/candidate FP pair is proved independently in Rewrite. -/
namespace VeriTile.Bench.Repairs.DropoutTriton.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.DropoutTriton
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

abbrev castBackend := @VeriTile.Bench.TritonBenchG.VectorAddition.TypedFloat.castBackend

theorem legacy_original (X K O : RegionName) (n B : Nat) (p : ℝ) :
    LegacyPastedSurface.sourceKernel X K O n B p = TypedSource.source .fp32 .int32 Bool.false X K O n B p := rfl

theorem legacy_commuted (X K O : RegionName) (n B : Nat) (p : ℝ) :
    LegacyPastedSurface.commutedSourceKernel X K O n B p = TypedSource.source .fp32 .int32 Bool.true X K O n B p := rfl

/-- Whole optional state equality under the existing identity-cast backend. -/
theorem original {A : Type} [Inhabited A] (M : Algebra A)
    (X K O : RegionName) (n B : Nat) (p : ℝ) (s : State A) :
    FP.Structural.exec (castBackend M) (sourceKernel X K O n B p) s =
      FP.Structural.exec (castBackend M) (TypedSource.source .fp32 .int32 Bool.false X K O n B p) s := by
  rw [←legacy_original]
  simp [sourceKernel,LegacyPastedSurface.sourceKernel,FP.Structural.exec,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,numericLt,numericEq,natLt,
    numeric,ofFloat,toFloat,ComputeDType.eraseDType,castBackend,
    VeriTile.Bench.TritonBenchG.VectorAddition.TypedFloat.castBackend,store,bop,VeriTile.Triton.DSL.TypedStores.fp32CallValue,
    Broadcast.leftIndex,Broadcast.rightIndex]

theorem original_real_execution (X K O : RegionName) (n B : Nat) (p : ℝ) (s : BlockState) :
    VeriTile.Triton.exec (sourceKernel X K O n B p).eraseDType.toAlgKernel s =
      VeriTile.Triton.exec (TypedSource.source .fp32 .int32 Bool.false X K O n B p).eraseDType.toAlgKernel s := by
  rw [←legacy_original]
  simp [sourceKernel,LegacyPastedSurface.sourceKernel,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    NumericDType.eraseDType,ComparableDType.eraseDType,FloatDType.toTileDType,
    VeriTile.Triton.eraseDType,Except.bind,bind,
    VeriTile.Triton.exec,stepStmts,stepStmt,evalOp.eq_def,
    Tile.bop,Tile.cop,VeriTile.Triton.DSL.TypedStores.fp32CallValue]

theorem original_real_trace (X K O : RegionName) (n B : Nat) (p : ℝ) (bounds : RegionBounds) (s : BlockState) :
    Kernel.TraceSafe bounds (sourceKernel X K O n B p).eraseDType.toAlgKernel s ↔
      Kernel.TraceSafe bounds (TypedSource.source .fp32 .int32 Bool.false X K O n B p).eraseDType.toAlgKernel s := by
  rw [←legacy_original]
  simp [Kernel.TraceSafe,Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt,sourceKernel,LegacyPastedSurface.sourceKernel,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    NumericDType.eraseDType,ComparableDType.eraseDType,FloatDType.toTileDType,
    VeriTile.Triton.eraseDType,Except.bind,bind,
    VeriTile.Triton.exec,stepStmts,stepStmt,evalOp.eq_def,
    Tile.bop,Tile.cop,VeriTile.Triton.DSL.TypedStores.fp32CallValue]

/-- Whole optional state equality under the existing identity-cast backend. -/
theorem commuted {A : Type} [Inhabited A] (M : Algebra A)
    (X K O : RegionName) (n B : Nat) (p : ℝ) (s : State A) :
    FP.Structural.exec (castBackend M) (commutedSourceKernel X K O n B p) s =
      FP.Structural.exec (castBackend M) (TypedSource.source .fp32 .int32 Bool.true X K O n B p) s := by
  rw [←legacy_commuted]
  simp [commutedSourceKernel,LegacyPastedSurface.commutedSourceKernel,FP.Structural.exec,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,numericLt,numericEq,natLt,
    numeric,ofFloat,toFloat,ComputeDType.eraseDType,castBackend,
    VeriTile.Bench.TritonBenchG.VectorAddition.TypedFloat.castBackend,store,bop,VeriTile.Triton.DSL.TypedStores.fp32CallValue,
    Broadcast.leftIndex,Broadcast.rightIndex]

theorem commuted_real_execution (X K O : RegionName) (n B : Nat) (p : ℝ) (s : BlockState) :
    VeriTile.Triton.exec (commutedSourceKernel X K O n B p).eraseDType.toAlgKernel s =
      VeriTile.Triton.exec (TypedSource.source .fp32 .int32 Bool.true X K O n B p).eraseDType.toAlgKernel s := by
  rw [←legacy_commuted]
  simp [commutedSourceKernel,LegacyPastedSurface.commutedSourceKernel,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    NumericDType.eraseDType,ComparableDType.eraseDType,FloatDType.toTileDType,
    VeriTile.Triton.eraseDType,Except.bind,bind,
    VeriTile.Triton.exec,stepStmts,stepStmt,evalOp.eq_def,
    Tile.bop,Tile.cop,VeriTile.Triton.DSL.TypedStores.fp32CallValue]

theorem commuted_real_trace (X K O : RegionName) (n B : Nat) (p : ℝ) (bounds : RegionBounds) (s : BlockState) :
    Kernel.TraceSafe bounds (commutedSourceKernel X K O n B p).eraseDType.toAlgKernel s ↔
      Kernel.TraceSafe bounds (TypedSource.source .fp32 .int32 Bool.true X K O n B p).eraseDType.toAlgKernel s := by
  rw [←legacy_commuted]
  simp [Kernel.TraceSafe,Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt,commutedSourceKernel,LegacyPastedSurface.commutedSourceKernel,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    NumericDType.eraseDType,ComparableDType.eraseDType,FloatDType.toTileDType,
    VeriTile.Triton.eraseDType,Except.bind,bind,
    VeriTile.Triton.exec,stepStmts,stepStmt,evalOp.eq_def,
    Tile.bop,Tile.cop,VeriTile.Triton.DSL.TypedStores.fp32CallValue]

/-- Complete equality after arbitrary flat allocation translation; no disjointness
or extra pointer premise is needed for this source-to-source connection. -/
theorem original_flat_execution (X K O : RegionName) (n B : Nat) (p : ℝ)
    (a : FlatAlloc) (s : BlockState) :
    VeriTile.Triton.exec (a.flattenKernel (sourceKernel X K O n B p).eraseDType.toAlgKernel) s =
      VeriTile.Triton.exec (a.flattenKernel (TypedSource.source .fp32 .int32 Bool.false X K O n B p).eraseDType.toAlgKernel) s := by
  rw [←legacy_original]
  simp [sourceKernel,LegacyPastedSurface.sourceKernel,FlatAlloc.flattenKernel,
    FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,FlatAlloc.flattenAccess,FlatAlloc.flattenMask,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    NumericDType.eraseDType,ComparableDType.eraseDType,FloatDType.toTileDType,
    VeriTile.Triton.eraseDType,Except.bind,bind,
    VeriTile.Triton.exec,stepStmts,stepStmt,evalOp.eq_def,
    Tile.bop,Tile.cop,VeriTile.Triton.DSL.TypedStores.fp32CallValue]

/-- Complete equality after arbitrary flat allocation translation; no disjointness
or extra pointer premise is needed for this source-to-source connection. -/
theorem commuted_flat_execution (X K O : RegionName) (n B : Nat) (p : ℝ)
    (a : FlatAlloc) (s : BlockState) :
    VeriTile.Triton.exec (a.flattenKernel (commutedSourceKernel X K O n B p).eraseDType.toAlgKernel) s =
      VeriTile.Triton.exec (a.flattenKernel (TypedSource.source .fp32 .int32 Bool.true X K O n B p).eraseDType.toAlgKernel) s := by
  rw [←legacy_commuted]
  simp [commutedSourceKernel,LegacyPastedSurface.commutedSourceKernel,FlatAlloc.flattenKernel,
    FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,FlatAlloc.flattenAccess,FlatAlloc.flattenMask,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    NumericDType.eraseDType,ComparableDType.eraseDType,FloatDType.toTileDType,
    VeriTile.Triton.eraseDType,Except.bind,bind,
    VeriTile.Triton.exec,stepStmts,stepStmt,evalOp.eq_def,
    Tile.bop,Tile.cop,VeriTile.Triton.DSL.TypedStores.fp32CallValue]

#axiomsClean original
#axiomsClean original_real_execution
#axiomsClean original_real_trace
#axiomsClean commuted
#axiomsClean commuted_real_execution
#axiomsClean commuted_real_trace
#axiomsClean original_flat_execution
#axiomsClean commuted_flat_execution
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DropoutTriton.Source
