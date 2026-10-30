import bench.tritonbench_g.attention_llama.Proofs.SourceExecution
import bench.tritonbench_g.attention_llama.Proofs.Mathematical_FP8RealSource

namespace VeriTile.Bench.Repairs.AttentionLlama.FP8Source
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionLlama
open RealKernel FP8RealSource
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000

def loopBody (fused : Bool) (scale : ℝ) (skn svk N SP BM BN BD : Nat)
    (causal : Bool) : List Stmt :=
  (SourceExecution.loopPrefix scale N BM BN BD).flatMap (lowerLoad fused) ++
    .ifThen (.constBool causal) [alCausalWhereStmt SP BM BN] ::
      (FP8RealSource.sourceTail skn svk N BM BN BD Bool.false).flatMap (lowerLoad fused)

theorem lowerList_fuse (code : List Stmt) (s : BlockState) :
    stepStmts (code.flatMap (lowerLoad Bool.false)) s =
      stepStmts (code.flatMap (lowerLoad Bool.true)) s := by
  induction code generalizing s with
  | nil => rfl
  | cons stmt rest ih =>
    simp only [List.flatMap_cons, ProgramRewrite.append]
    have hf : stepStmts (lowerLoad Bool.false stmt) s = stepStmts (lowerLoad Bool.true stmt) s := by
      cases h : byteLoad? stmt with
      | none => simp [lowerLoad,h]
      | some load =>
        simp only [lowerLoad,h,Bool.false_eq_true,↓reduceIte]
        exact FP8RealLoads.fuse_conversion ..
    rw [hf]
    congr 1
    funext st
    exact ih st

theorem loop_fuse (scale : ℝ) (skn svk N SP BM BN BD : Nat)
    (causal : Bool) (s : BlockState) :
    stepStmts (loopBody Bool.false scale skn svk N SP BM BN BD causal) s =
      stepStmts (loopBody Bool.true scale skn svk N SP BM BN BD causal) s := by
  simp only [loopBody,ProgramRewrite.append,lowerList_fuse]
  congr 1
  funext st
  simp only [stepStmts,lowerList_fuse]

section
variable (Q K V Out : RegionName) (scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD : Nat) (causal : Bool)

def body (fused : Bool) : List Stmt :=
  alPreLoopG Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn NH H N BM BN BD ++
    .ifThen (.constBool causal) [alBnEndCausalStmt SP BN] ::
    .forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "block_n_end") (.constNat BN)
      (loopBody fused scale skn svk N SP BM BN BD causal) ::
      alPostLoopG Out soz soh som son H BM BD

theorem source_body :
    (TypedSource.repairedFP8 Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal).eraseDType.toAlgKernel.body =
      body Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal Bool.false := by
  simp only [TypedSource.repairedFP8,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.surfaceBody,ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Kernel.eraseDType,
    Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType,MemAccess.eraseDType,
    MaskOpt.eraseDType,VeriTile.Triton.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,
    List.map_cons,List.map_nil]
  simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,List.map_cons,List.map_nil]
  rfl

theorem projection_supported :
    (TypedSource.repairedFP8 Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal).eraseDType.toAlgorithm? =
      .ok (TypedSource.repairedFP8 Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal).eraseDType.toAlgKernel := by
  have h : (TypedSource.repairedFP8 Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal).toAlgorithm? =
      .ok (TypedSource.repairedFP8 Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal).toAlgKernel := rfl
  simp only [ComputeKernel.eraseDType,h]
  simp

theorem fused_execution (s : BlockState) :
    exec (TypedSource.repairedFP8 Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal).eraseDType.toAlgKernel s =
      stepStmts (body Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal Bool.true) s := by
  simp only [exec,source_body,body,ProgramRewrite.append,stepStmts,
    ProgramRewrite.dynamic_range_congr _ _ _ _ _ _ (loop_fuse scale skn svk N SP BM BN BD causal)]

end
end VeriTile.Bench.Repairs.AttentionLlama.FP8Source
