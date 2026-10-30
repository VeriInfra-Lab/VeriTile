import bench.tritonbench_g.matmul_triton2.Proofs.FPSource
import bench.tritonbench_g.matmul_triton2.MatmulTriton2
namespace VeriTile.Bench.TritonBenchG.MatmulTriton2.FPValue
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def legacySource (p : Parameters) : ComputeKernel :=
  matmul_triton2_surface p.input p.weight p.output p.rows p.columns (inner p)
    p.inputRow p.inputInner p.weightInner p.weightColumn p.outputRow p.outputColumn
    p.blockRows p.blockColumns p.blockInner p.groupRows

/-- Full original control/address/mask/dot-options projection. The explicit
same-format cast materializes the frontend's dtype boundary, and is identity
in the new floating execution model. -/
theorem source_projection (p : Parameters) :
    (source p).eraseDType.surfaceBody=(legacySource p).eraseDType.surfaceBody := by
  simp [source,typed,legacySource,matmul_triton2_surface,
    ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType,
    MemAccess.eraseDType,MaskOpt.eraseDType,ComputeKernel.fromKernelBody,
    ComputeKernel.surfaceBody,ComputeDType.eraseDType,Except.bind,bind,pure,
    VeriTile.Triton.eraseDType,FloatDType.numericDType,NumericDType.eraseDType]
  all_goals try simp only [Op.eraseDType.eq_def]
  all_goals simp [FloatDType.toTileDType,VeriTile.Triton.eraseDType]

#axiomsClean source_projection
end
end VeriTile.Bench.TritonBenchG.MatmulTriton2.FPValue
