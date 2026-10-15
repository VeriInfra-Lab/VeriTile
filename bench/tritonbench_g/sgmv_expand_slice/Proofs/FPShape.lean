import bench.tritonbench_g.sgmv_expand_slice.Proofs.FPSource
import bench.tritonbench_g.sgmv_expand_slice.SgmvExpandSlice
namespace VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 26000
set_option linter.unusedSimpArgs false
/-- Complete original accepted active body, including every K iteration and
address expression. Casts are separately preserved by the typed value proof. -/
theorem source_projection (fmt : Storage) (out : Output) (p : Parameters) :
    (source fmt out p).eraseDType.surfaceBody =
      (sgmv_expand_slice_surface p.input p.weight p.output p.starts p.lengths p.adapters p.columns p.inner
        p.inputRowStride p.inputInnerStride p.adapterStride p.weightColumnStride p.weightInnerStride
        p.outputRowStride p.outputColumnStride p.sliceOffset p.blockRows p.blockColumns p.blockInner).eraseDType.surfaceBody := by
  cases fmt <;> cases out <;>
    simp [source,halfTyped,bfloatTyped,outputType,sgmv_expand_slice_surface,
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
#axiomsClean source_projection
end
end VeriTile.Bench.TritonBenchG.SgmvExpandSlice.FPValue
