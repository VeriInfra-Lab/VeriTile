import bench.tritonbench_g.rotary_transform.Proofs.FPRow
namespace VeriTile.Bench.TritonBenchG.RotaryTransform.FPRow
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
@[simp] theorem erase_real_cast {shape : TileShape} (e : Op .real shape) :
    (Op.castFloat .real .real e).eraseDType = e.eraseDType := by
  rw [Op.eraseDType_castFloat]
  rfl

/-- Restoring the four Python fp32 register casts preserves the entire
previously selected paired row and either metadata branch after erasure. -/
theorem source_projection (k : Kind) (p : Parameters) :
    (source k p).eraseDType.surfaceBody = (legacySource k p).eraseDType.surfaceBody := by
  cases k
  all_goals simp [source, legacySource, rowTyped, metaTyped, rotary_kernel_o0o1_row, rotary_meta_row,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
    MemAccess.eraseDType, MaskOpt.eraseDType, ComputeKernel.fromKernelBody,
    ComputeKernel.surfaceBody, ComputeDType.eraseDType, Except.bind, bind, pure, erase_real_cast]
  all_goals repeat' constructor
  all_goals conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals simp [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType]
#axiomsClean source_projection
end
end VeriTile.Bench.TritonBenchG.RotaryTransform.FPRow
