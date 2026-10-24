import bench.tritonbench_g.int8_matmul_quantization.Proofs.QuantizeSource
import bench.tritonbench_g.int8_matmul_quantization.Proofs.QuantizeDomainCompat
namespace VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPQuantize
open VeriTile.Triton
set_option maxHeartbeats 8000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

theorem prelude_projection (p : Parameters) :
    (ComputeKernel.mk [] [] (prelude p)).eraseDType.toAlgKernel.body =
      qPrologue p.input p.output p.rows p.inputRowStride p.inputColumnStride
        p.outputRowStride p.outputColumnStride p.blockRows p.blockColumns := by
  simp [prelude, source, qPrologue, ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    FloatDType.toTileDType, NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    VeriTile.Triton.eraseDType, ComputeDType.eraseDType, Region.cast, Except.bind, bind]
  try rfl

theorem first_load_projection (p : Parameters) :
    (ComputeKernel.mk [] [] (firstBody p)).eraseDType.toAlgKernel.body.take 1 =
      (qLoop1Body p.columns p.inputColumnStride p.blockRows p.blockColumns).take 1 := by
  simp [firstBody, source, qLoop1Body, qFpaLoadOp, qKMaskOp, ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    FloatDType.toTileDType, NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    VeriTile.Triton.eraseDType, ComputeDType.eraseDType, Region.cast, Except.bind, bind]
  try rfl

theorem second_load_projection (p : Parameters) :
    (ComputeKernel.mk [] [] (secondBody p)).eraseDType.toAlgKernel.body.take 1 =
      (qLoop2Body p.columns p.inputColumnStride p.outputColumnStride p.blockRows p.blockColumns).take 1 := by
  simp [secondBody, source, qLoop2Body, qFpaLoadOp, qKMaskOp, ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    FloatDType.toTileDType, NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    VeriTile.Triton.eraseDType, ComputeDType.eraseDType, Region.cast, Except.bind, bind]
  try rfl

theorem first_pointer_projection (p : Parameters) :
    (ComputeKernel.mk [] [] (firstBody p)).eraseDType.toAlgKernel.body.drop 2 =
      (qLoop1Body p.columns p.inputColumnStride p.blockRows p.blockColumns).drop 2 := by
  simp [firstBody, source, qLoop1Body, ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    FloatDType.toTileDType, NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    VeriTile.Triton.eraseDType, ComputeDType.eraseDType, Region.cast, Except.bind, bind]
  try rfl

theorem second_pointers_projection (p : Parameters) :
    (ComputeKernel.mk [] [] (secondBody p)).eraseDType.toAlgKernel.body.drop 3 =
      (qLoop2Body p.columns p.inputColumnStride p.outputColumnStride p.blockRows p.blockColumns).drop 3 := by
  simp [secondBody, source, qLoop2Body, ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    FloatDType.toTileDType, NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    VeriTile.Triton.eraseDType, ComputeDType.eraseDType, Region.cast, Except.bind, bind]
  try rfl

/-- The pointwise maximum in the first pass is the true primitive in the
parsed source, including the fp32 context and the independent axis-1 reduction. -/
theorem first_maximum_primitive (p : Parameters) :
    (firstBody p)[1]? = some (.assign .real [p.blockRows] "a_max"
      (.compute (.alg .fp32 (.maximumFloat .real (.consSame .nil)
        (.ref .real [p.blockRows] "a_max")
        (.reduceMax (shape := [p.blockRows,p.blockColumns]) ⟨1,by simp⟩ Bool.false
          (.absFloat .real (.ref .real [p.blockRows,p.blockColumns] "fpa"))))))) := rfl

/-- Actual callback selection is independent of the fallback's comparison. -/
theorem maximum_callback (rest : FP.Structural.Algebra Word) (cfg : FP.Config)
    (u : Nat → Word → Int) (a b : Word) :
    (backend rest cfg u).binary (some .fp32) .real .max a b = pairMaximum a b := by
  simp [backend, FP.Software.fp32Context]

/-- The former comparison/select lowering fails on a NaN right operand.
Both operand orders and the explicit all-NaN software policy are checked. -/
theorem maximum_nan_regression :
    pairMaximum (FP.Value.ofBits .fp32 0x3f800000) (FP.Value.qNaN .fp32) =
      FP.Value.ofBits .fp32 0x3f800000 ∧
    pairMaximum (FP.Value.qNaN .fp32) (FP.Value.ofBits .fp32 0x3f800000) =
      FP.Value.ofBits .fp32 0x3f800000 ∧
    pairMaximum (FP.Value.qNaN .fp32) (FP.Value.qNaN .fp32) = FP.Value.qNaN .fp32 ∧
    (if FP.lt .ieeeValue (FP.Value.qNaN .fp32) (FP.Value.ofBits .fp32 0x3f800000)
      then FP.Value.ofBits .fp32 0x3f800000 else FP.Value.qNaN .fp32) ≠
        pairMaximum (FP.Value.ofBits .fp32 0x3f800000) (FP.Value.qNaN .fp32) := by
  decide +kernel

#axiomsClean first_maximum_primitive
#axiomsClean maximum_callback
#axiomsClean maximum_nan_regression

#axiomsClean prelude_projection
#axiomsClean first_load_projection
#axiomsClean second_load_projection
#axiomsClean first_pointer_projection
#axiomsClean second_pointers_projection
end VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPQuantize
