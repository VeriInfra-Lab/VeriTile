import bench.tritonbench_g.int8_matmul_quantization.Proofs.MatmulSource
namespace VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPMatmul
open VeriTile.Triton
set_option maxHeartbeats 10000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def rawBody (p : Parameters) : List ComputeStmt :=
  match (typed p).surfaceBody[20]? with | some (ComputeStmt.forRange _ _ _ _ b) => b | _ => []
/-- Width materialization changes only the two signed loads and the int32
accumulator assignment. All five source statements remain in source order. -/
theorem body_materialization (p : Parameters) : body p = (rawBody p).map widths := by rfl
theorem body_length (p : Parameters) : (body p).length=5 := by rfl

theorem grouped_address_projection (p : Parameters) :
    (ComputeKernel.mk [] [] ((prelude p).take 17)).eraseDType.toAlgKernel.body =
      mmPreLoopScalars p.rows p.columns p.blockRows p.blockColumns p.groupRows ++
      (mmPreLoopTiles p.input p.inputScale p.weight p.weightScale p.rows p.columns
        p.inputRowStride p.inputInnerStride p.inputScaleStride p.weightInnerStride
        p.weightColumnStride p.weightScaleStride p.blockRows p.blockColumns p.blockInner).take 7 := by
  simp [prelude, source, typed, mmPreLoopScalars, mmPreLoopTiles, ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    FloatDType.toTileDType, NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    VeriTile.Triton.eraseDType, ComputeDType.eraseDType, Region.cast, Except.bind, bind]
  try rfl

theorem loop_projection (p : Parameters) :
    (ComputeKernel.mk [] [] (rawBody p)).eraseDType.toAlgKernel.body =
      mmLoopBody p.inner p.inputInnerStride p.weightInnerStride p.blockRows p.blockColumns p.blockInner := by
  simp [rawBody, typed, mmLoopBody, mmAMaskOp, mmBMaskOp, ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    FloatDType.toTileDType, NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    VeriTile.Triton.eraseDType, ComputeDType.eraseDType, Region.cast, Except.bind, bind]
  try rfl

theorem store_address_projection (p : Parameters) :
    (ComputeKernel.mk [] [] ((epilogue p).drop 1)).toAlgKernel.body =
      (mmPostLoop p.output p.rows p.columns p.outputRowStride p.outputColumnStride p.blockRows p.blockColumns).drop 1 := by
  simp [epilogue, source, typed, mmPostLoop, ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    FloatDType.toTileDType, NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    VeriTile.Triton.eraseDType, ComputeDType.eraseDType, Region.cast, Except.bind, bind]
  try rfl

#axiomsClean body_materialization
#axiomsClean body_length
#axiomsClean grouped_address_projection
#axiomsClean loop_projection
#axiomsClean store_address_projection
end VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPMatmul
