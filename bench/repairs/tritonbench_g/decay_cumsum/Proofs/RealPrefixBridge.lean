import bench.repairs.tritonbench_g.decay_cumsum.Proofs.RealSourceBridge

namespace VeriTile.Bench.Repairs.DecayCumsum.RealPrefixBridge
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DecayCumsum
open RealBodyBridge
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

def legacyPrefix (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) : List Stmt :=
  (LegacySurface.source kind h p scale).eraseDType.toAlgKernel.body.take
    (RealSourceBridge.prefixCount kind-2)

theorem forward (A : FlatAlloc) (h : Call) (p : Phase) (scale : ℝ) (u : BlockState) :
    (stepStmts (A.flattenStmts (legacyPrefix .forward h p scale)) u).map norm =
      (stepStmts (A.flattenStmts (RealSourceBridge.prelude .forward h p scale)) u).map norm := by
  simp only [legacyPrefix,LegacySurface.source,Original.forward,TypedSource.forward,TypedSource.inv_ln2,
    RealSourceBridge.prelude,RealSourceBridge.prefixCount,Kernels.source,Kernels.quotedForward,
    Bool.false_eq_true,↓reduceIte,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,Except.bind,bind,eraseDType_float]
  simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType]
  simp only [List.take,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,
    List.cons_append,List.nil_append,stepStmts,stepStmt,evalOp,Option.bind,Option.map,Tile.scalar_data]
  simp [-Tile.scalar_eta,BlockState.setReg,Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.scalar,Tile.vec,
    NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
  simp only [← Int.natCast_mul,← Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt]
  apply BlockState.ext
  · intros; rfl
  · intro d sh n
    by_cases hd : RealBodyBridge.dead n
    · simp only [RealBodyBridge.norm,hd,if_true]
    · have nk : n≠"i_k" := by intro hn; apply hd; simp [RealBodyBridge.dead,hn]
      have nc : n≠"i_c" := by intro hn; apply hd; simp [RealBodyBridge.dead,hn]
      have nb : n≠"i_bh" := by intro hn; apply hd; simp [RealBodyBridge.dead,hn]
      simp only [RealBodyBridge.norm,hd,nk,nc,nb,if_false]
      rfl
  · intros; rfl
  · intros; rfl
  · intros; rfl

theorem prepare (A : FlatAlloc) (h : Call) (p : Phase) (scale : ℝ) (positive : 0<p.span) (u : BlockState) :
    (stepStmts (A.flattenStmts (legacyPrefix .prepare h p scale)) u).map norm =
      (stepStmts (A.flattenStmts (RealSourceBridge.prelude .prepare h p scale)) u).map norm := by
  have time (n : Nat) : ((n*p.span+p.span : Nat) : Int)-1 = (n*p.span+p.span-1 : Nat) := by
    symm
    exact Int.ofNat_sub (by omega)
  simp only [legacyPrefix,LegacySurface.source,Original.prepare,TypedSource.prepare,TypedSource.inv_ln2,
    RealSourceBridge.prelude,RealSourceBridge.prefixCount,Kernels.source,Kernels.quotedPrepare,
    Bool.false_eq_true,↓reduceIte,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,Except.bind,bind,eraseDType_float]
  simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType]
  simp only [List.take,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,
    List.cons_append,List.nil_append,stepStmts,stepStmt,evalOp,Option.bind,Option.map,Tile.scalar_data]
  simp [-Tile.scalar_eta,BlockState.setReg,Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.scalar,Tile.vec,
    NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
  simp only [← Int.natCast_mul,← Int.natCast_add,time,Int.toNat_natCast,Int.ofNat_lt]
  apply BlockState.ext
  · intros; rfl
  · intro d sh n
    by_cases hd : RealBodyBridge.dead n
    · simp only [RealBodyBridge.norm,hd,if_true]
    · have nk : n≠"i_k" := by intro hn; apply hd; simp [RealBodyBridge.dead,hn]
      have nc : n≠"i_c" := by intro hn; apply hd; simp [RealBodyBridge.dead,hn]
      have nb : n≠"i_bh" := by intro hn; apply hd; simp [RealBodyBridge.dead,hn]
      simp only [RealBodyBridge.norm,hd,nk,nc,nb,if_false]
      rfl
  · intros; rfl
  · intros; rfl
  · intros; rfl

theorem backward (A : FlatAlloc) (h : Call) (p : Phase) (scale : ℝ) (positive : 0<p.span) (u : BlockState) :
    (stepStmts (A.flattenStmts (legacyPrefix .backward h p scale)) u).map norm =
      (stepStmts (A.flattenStmts (RealSourceBridge.prelude .backward h p scale)) u).map norm := by
  have time (n : Nat) : ((n*p.span+p.span : Nat) : Int)-1 = (n*p.span+p.span-1 : Nat) := by
    symm
    exact Int.ofNat_sub (by omega)
  simp only [legacyPrefix,LegacySurface.source,Original.backward,TypedSource.backward,TypedSource.inv_ln2,
    RealSourceBridge.prelude,RealSourceBridge.prefixCount,Kernels.source,Kernels.quotedBackward,
    Bool.false_eq_true,↓reduceIte,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,Except.bind,bind,eraseDType_float]
  simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType]
  simp only [List.take,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,
    List.cons_append,List.nil_append,stepStmts,stepStmt,evalOp,Option.bind,Option.map,Tile.scalar_data]
  simp [-Tile.scalar_eta,BlockState.setReg,Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.scalar,Tile.vec,
    NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
  simp only [← Int.natCast_mul,← Int.natCast_add,time,Int.toNat_natCast,Int.ofNat_lt]
  apply BlockState.ext
  · intros; rfl
  · intro d sh n
    by_cases hd : RealBodyBridge.dead n
    · simp only [RealBodyBridge.norm,hd,if_true]
    · have nk : n≠"i_k" := by intro hn; apply hd; simp [RealBodyBridge.dead,hn]
      have nc : n≠"i_c" := by intro hn; apply hd; simp [RealBodyBridge.dead,hn]
      have nb : n≠"i_bh" := by intro hn; apply hd; simp [RealBodyBridge.dead,hn]
      simp only [RealBodyBridge.norm,hd,nk,nc,nb,if_false]
      rfl
  · intros; rfl
  · intros; rfl
  · intros; rfl

end
end VeriTile.Bench.Repairs.DecayCumsum.RealPrefixBridge
