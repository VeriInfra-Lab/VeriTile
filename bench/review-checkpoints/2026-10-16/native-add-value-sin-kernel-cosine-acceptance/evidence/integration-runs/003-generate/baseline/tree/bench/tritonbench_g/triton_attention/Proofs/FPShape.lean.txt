import bench.tritonbench_g.triton_attention.Proofs.FPForwardReadback
import bench.tritonbench_g.triton_attention.Proofs.FPBackwardReference
import bench.tritonbench_g.triton_attention.TritonAttention
namespace VeriTile.Bench.TritonBenchG.TritonAttention.FPValue
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

/-- Erasure checks only the old source projection. It supplies no floating law. -/
theorem preprocess_projection (p : PreParameters) :
    (preSource p).eraseDType.surfaceBody =
      (triton_attention_bwd_preprocess p.output p.gradient p.denominator p.newGradient p.delta p.rows p.head).eraseDType.surfaceBody := by
  rcases p with ⟨dtype,o,g,l,ng,de,bm,bd⟩
  cases dtype <;> simp [preSource,preprocessTyped,triton_attention_bwd_preprocess,
    ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType,
    MemAccess.eraseDType,MaskOpt.eraseDType,ComputeKernel.fromKernelBody,
    ComputeKernel.surfaceBody,ComputeDType.eraseDType,Except.bind,bind,pure,
    VeriTile.Triton.eraseDType,FloatDType.numericDType,NumericDType.eraseDType]
  all_goals simp [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, eraseDType_float, FloatDType.toTileDType,
    VeriTile.Triton.eraseDType, Region.cast, NumericDType.eraseDType]

theorem forward_projection (p : ForwardParameters) (qz Z H : Nat) :
    (forwardSource p).eraseDType.surfaceBody =
      (triton_attention_fwd_kernel p.q p.k p.v p.denominator p.maximum p.output p.scale
        qz p.headStride p.head 1 qz p.headStride p.head 1 qz p.headStride p.head 1
        qz p.headStride p.head 1 Z H p.context p.rows p.blockM p.head p.blockN).eraseDType.surfaceBody := by
  rcases p with ⟨dtype,q,k,v,l,m,o,sc,hs,ctx,rows,bm,bn,bd⟩
  cases dtype <;> simp [forwardSource,forwardTyped,triton_attention_fwd_kernel,
    ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType,
    MemAccess.eraseDType,MaskOpt.eraseDType,ComputeKernel.fromKernelBody,
    ComputeKernel.surfaceBody,ComputeDType.eraseDType,Except.bind,bind,pure,
    VeriTile.Triton.eraseDType,FloatDType.numericDType,NumericDType.eraseDType]
  all_goals simp [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, eraseDType_float, FloatDType.toTileDType,
    VeriTile.Triton.eraseDType, Region.cast, NumericDType.eraseDType]

theorem backward_projection (p : BackwardParameters) (Out L : RegionName) (Z skz skh svz svh BN : Nat) :
    (backwardSource p).eraseDType.surfaceBody =
      (triton_attention_bwd_kernel p.q p.k p.v Out p.gradient p.dq p.dk p.dv L p.maximum p.delta p.scale
        p.batchStride p.headStride p.head 1 skz skh p.head 1 svz svh p.head 1
        Z p.heads (p.blockM*p.blocks) p.rows p.blocks p.blockM p.head BN).eraseDType.surfaceBody := by
  rcases p with ⟨dtype,q,k,v,g,dq,dk,dv,m,de,sc,bs,hs,heads,rows,bm,bd,nb⟩
  cases dtype <;> simp [backwardSource,backwardTyped,triton_attention_bwd_kernel,
    ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType,
    MemAccess.eraseDType,MaskOpt.eraseDType,ComputeKernel.fromKernelBody,
    ComputeKernel.surfaceBody,ComputeDType.eraseDType,Except.bind,bind,pure,
    VeriTile.Triton.eraseDType,FloatDType.numericDType,NumericDType.eraseDType]
  all_goals simp [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, eraseDType_float, FloatDType.toTileDType,
    VeriTile.Triton.eraseDType, Region.cast, NumericDType.eraseDType]

#axiomsClean preprocess_projection
#axiomsClean forward_projection
#axiomsClean backward_projection
end VeriTile.Bench.TritonBenchG.TritonAttention.FPValue
