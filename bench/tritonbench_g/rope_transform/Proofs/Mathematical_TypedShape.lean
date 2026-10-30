import bench.tritonbench_g.rope_transform.Proofs.Mathematical_TypedProof
import bench.tritonbench_g.rope_transform.Proofs.Mathematical_RopeTransform
namespace VeriTile.Bench.TritonBenchG.RopeTransform.Typed
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
/-- Full source shape, including both branches and all four stores. Erasure
checks address/control correspondence only; it is not numerical equivalence. -/
theorem source_shape (p : Parameters) :
    (kernel .fp32 .fp32 p).eraseDType.surfaceBody =
      (triton_rope_surface p.q p.k p.cos p.sin p.qStride p.kStride p.cosStride
        p.sinStride p.length 0 p.qHeads p.kHeads p.headDim p.paddedQHeads
        p.paddedKHeads p.paddedHeadDim 0 p.backward).eraseDType.surfaceBody := by
  simp [kernel, source, triton_rope_surface, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody, ComputeDType.eraseDType,
    Except.bind, bind, pure, FloatDType.toTileDType, FloatDType.numericDType]
  repeat' constructor
  all_goals conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals simp [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType]
  all_goals simp only [FloatDType.toTileDType, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, Op.eraseDType.eq_def, NumericDType.eraseDType, ComparableDType.eraseDType, VeriTile.Triton.eraseDType]
  all_goals rfl
#axiomsClean source_shape
end VeriTile.Bench.TritonBenchG.RopeTransform.Typed
