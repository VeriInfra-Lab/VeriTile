import bench.tritonbench_g.lightning_attention.Proofs.FPSource
import bench.tritonbench_g.lightning_attention.LightningAttention
namespace VeriTile.Bench.TritonBenchG.LightningAttention.FPValue
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

/-- Exact control/address correspondence to the entire old accepted stage.
Physical storage and register conversions are checked separately at value level. -/
theorem kv_source_projection (p : Parameters) :
    (kvSource p).eraseDType.surfaceBody =
      (lightning_attention_forward_kv_step_slice p.carry p.key p.value p.output p.keyDim p.rows p.columns).eraseDType.surfaceBody := by
  simp [kvSource,kvTyped,lightning_attention_forward_kv_step_slice,
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
#axiomsClean kv_source_projection

/-- Exact control/address correspondence to the entire old accepted stage.
Physical storage and register conversions are checked separately at value level. -/
theorem inter_source_projection (p : Parameters) :
    (interSource p).eraseDType.surfaceBody =
      (lightning_attention_forward_o_inter_dot_slice p.query p.carry p.output p.rows p.keyDim p.columns).eraseDType.surfaceBody := by
  simp [interSource,interTyped,lightning_attention_forward_o_inter_dot_slice,
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
#axiomsClean inter_source_projection

/-- Exact control/address correspondence to the entire old accepted stage.
Physical storage and register conversions are checked separately at value level. -/
theorem body_source_projection (p : Parameters) :
    (bodySource p).eraseDType.surfaceBody =
      (lightning_attention_forward_body_slice p.query p.key p.value p.carry p.output p.blockIndex p.length p.keyDim p.valueDim p.rows p.columns).eraseDType.surfaceBody := by
  simp [bodySource,bodyTyped,lightning_attention_forward_body_slice,
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
#axiomsClean body_source_projection

/-- Exact control/address correspondence to the entire old accepted stage.
Physical storage and register conversions are checked separately at value level. -/
theorem grad_source_projection (p : Parameters) :
    (gradSource p).eraseDType.surfaceBody =
      (lightning_attention_bwd_grad_store_slice p.query p.output p.length p.valueDim p.rows p.columns).eraseDType.surfaceBody := by
  simp [gradSource,gradTyped,lightning_attention_bwd_grad_store_slice,
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
#axiomsClean grad_source_projection
end
end VeriTile.Bench.TritonBenchG.LightningAttention.FPValue
