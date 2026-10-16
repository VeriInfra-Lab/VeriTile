import bench.tritonbench_g.matmul_tma.Proofs.FPSource
import bench.tritonbench_g.matmul_tma.MatmulTma
namespace VeriTile.Bench.TritonBenchG.MatmulTma.FPValue
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def legacySource (p : Parameters) (half : Bool) : ComputeKernel :=
  if half then
    matmul_tma_f16_surface p.a p.b p.c p.rows p.columns p.inner p.aRow p.aInner
      p.bInner p.bColumn p.cRow p.cColumn p.blockRows p.blockColumns p.blockInner
  else
    matmul_tma_f32_surface p.a p.b p.c p.rows p.columns p.inner p.aRow p.aInner
      p.bInner p.bColumn p.cRow p.cColumn p.blockRows p.blockColumns p.blockInner

/-- Both original constexpr branches have exactly the same control/address/dot
projection. This does not erase casts in the floating execution theorem. -/
theorem source_projection (p : Parameters) (half : Bool) :
    (source p half).eraseDType.surfaceBody = (legacySource p half).eraseDType.surfaceBody := by
  cases half <;>
    simp [source, legacySource, typed, outputDType, matmul_tma_f32_surface,
      matmul_tma_f16_surface, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, ComputeKernel.fromKernelBody,
      ComputeKernel.surfaceBody, ComputeDType.eraseDType, Except.bind, bind, pure,
      VeriTile.Triton.eraseDType, FloatDType.numericDType, NumericDType.eraseDType]
  all_goals try conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals try simp [Op.eraseDType, VeriTile.Triton.eraseDType]
  all_goals try simp only [Op.eraseDType.eq_def]
  all_goals rfl

#axiomsClean source_projection
end
end VeriTile.Bench.TritonBenchG.MatmulTma.FPValue
