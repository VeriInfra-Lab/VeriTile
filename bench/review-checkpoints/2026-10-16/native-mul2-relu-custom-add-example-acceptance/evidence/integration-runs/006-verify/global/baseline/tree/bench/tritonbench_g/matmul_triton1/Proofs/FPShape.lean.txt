import bench.tritonbench_g.matmul_triton1.Proofs.FPSource
import bench.tritonbench_g.matmul_triton1.MatmulTriton1
namespace VeriTile.Bench.TritonBenchG.MatmulTriton1.FPValue
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def legacySource (p : Parameters) : ComputeKernel :=
  matmul_triton1_surface p.input p.weight p.output p.rows (inner p) p.columns
    p.blockRows p.blockInner p.blockColumns

/-- Complete source bridge, including every K-loop iteration, dot options,
program-ID addressing and final store. The fp32→fp32 annotation materializes
a bit-preserving cast needed by the parameterized typed frontend. -/
theorem source_projection (p : Parameters) :
    (source p).eraseDType.surfaceBody = (legacySource p).eraseDType.surfaceBody := by
  simp [source, typed, legacySource, matmul_triton1_surface,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
    MemAccess.eraseDType, MaskOpt.eraseDType, ComputeKernel.fromKernelBody,
    ComputeKernel.surfaceBody, ComputeDType.eraseDType, Except.bind, bind, pure,
    VeriTile.Triton.eraseDType, FloatDType.numericDType, NumericDType.eraseDType]
  all_goals try simp only [Op.eraseDType.eq_def]
  all_goals simp [FloatDType.toTileDType, VeriTile.Triton.eraseDType]

#axiomsClean source_projection
end
end VeriTile.Bench.TritonBenchG.MatmulTriton1.FPValue
