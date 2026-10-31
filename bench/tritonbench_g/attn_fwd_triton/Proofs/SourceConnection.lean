import bench.tritonbench_g.attn_fwd_triton.Proofs.SourceFragments

/- Exact execution connection between the repair quotation and its proof view.
The view factors a literal branch; every selected instruction and bound is retained. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified
open VeriTile.Triton
set_option maxRecDepth 32000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

def actualCode (BM BN BD : Nat) (p : Parameters) (r : Bool) : List Stmt :=
  (source BM BN BD p r).eraseDType.toAlgKernel.body

theorem actual_branch (BM BN BD : Nat) (p : Parameters) (r : Bool) :
    ControlView.AlgorithmBranch 32 (actualCode BM BN BD p r) := by
  simp only [actualCode,source,TypedSource.repaired,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,
    ComputeKernel.surfaceBody,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,FloatDType.toTileDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,
    ComputeStmt.listToAlgorithm?_map_alg]
  repeat first | apply ControlView.AlgorithmBranch.next | exact ControlView.AlgorithmBranch.here _ _ _ _

theorem code_run (BM BN BD : Nat) (p : Parameters) (r : Bool) (s : BlockState) :
    stepStmts (code BM BN BD p r) s=stepStmts (actualCode BM BN BD p r) s :=
  ControlView.algorithm_run 32 6 _ (actual_branch BM BN BD p r) s

theorem actual_execution (BM BN BD : Nat) (p : Parameters) (r : Bool)
    (s t : BlockState) (h : stepStmts (code BM BN BD p r) s=some t) :
    stepStmts (actualCode BM BN BD p r) s=some t := by
  rw [←code_run]
  exact h

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified
