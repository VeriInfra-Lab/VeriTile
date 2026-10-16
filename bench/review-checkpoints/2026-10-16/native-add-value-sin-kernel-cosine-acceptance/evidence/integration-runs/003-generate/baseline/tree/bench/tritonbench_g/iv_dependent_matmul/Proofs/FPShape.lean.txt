import bench.tritonbench_g.iv_dependent_matmul.Proofs.FPReference
import bench.tritonbench_g.iv_dependent_matmul.IvDependentMatmul
namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false

theorem source_projection (p : Parameters) :
    (source .source p).eraseDType.surfaceBody =
      (iv_dependent_matmul_pre_load_surface p.input p.weight p.output p.rows p.columns p.inner
        p.inputRowStride p.inputInnerStride p.weightInnerStride p.weightColumnStride
        p.outputRowStride p.outputColumnStride p.blockRows p.blockColumns p.blockInner).eraseDType.surfaceBody := by
  simp [source,typed,iv_dependent_matmul_pre_load_surface,
      ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
      Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType,
      MemAccess.eraseDType,MaskOpt.eraseDType,ComputeKernel.fromKernelBody,
      ComputeKernel.surfaceBody,ComputeDType.eraseDType,Except.bind,bind,pure,
      VeriTile.Triton.eraseDType,FloatDType.numericDType,NumericDType.eraseDType]
  all_goals repeat' constructor
  all_goals try conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals try simp [Op.eraseDType,VeriTile.Triton.eraseDType]
  all_goals try simp only [Op.eraseDType.eq_def]
  all_goals try simp only [NumericDType.eraseDType,eq_mp_eq_cast,eq_mpr_eq_cast,cast_eq]
  all_goals repeat' constructor
  all_goals rfl

theorem compiled_dot_node (p : Parameters) :
    (body .accumulator p)[4]? = some (.assign .real [p.blockRows,p.blockColumns] "accumulator"
      (.compute (.alg .fp32 (.dotAcc .fp32 .fp32 .real (batch := [])
        (.ref .fp32 [p.blockRows,p.blockInner] "a")
        (.ref .fp32 [p.blockInner,p.blockColumns] "b")
        (.ref .real [p.blockRows,p.blockColumns] "accumulator") {})))) := by rfl
#axiomsClean source_projection
#axiomsClean compiled_dot_node
end
end VeriTile.Bench.TritonBenchG.IvDependentMatmul.FPValue
