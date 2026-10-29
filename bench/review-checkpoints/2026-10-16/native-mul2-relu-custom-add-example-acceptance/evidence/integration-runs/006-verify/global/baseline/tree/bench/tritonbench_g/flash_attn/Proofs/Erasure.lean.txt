import bench.tritonbench_g.flash_attn.Proofs.NormalizedSource
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FlashAttn.Erasure
open VeriTile.Triton TypedSource
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem body (input output : Storage) (Q K V L O : RegionName) (sm_scale : ℝ)
    (stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
      stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
      stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
      BS HEAD SEQLEN BLOCK_M DIM BLOCK_N : Nat) (causal : Bool) (swapped : Bool) :
    (kernel input output swapped Q K V L O sm_scale stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
    stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
    stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
    stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
    BS HEAD SEQLEN BLOCK_M DIM BLOCK_N causal).eraseDType.toAlgKernel =
    (kernel .fp32 .fp32 swapped Q K V L O sm_scale stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
    stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
    stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
    stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
    BS HEAD SEQLEN BLOCK_M DIM BLOCK_N causal).eraseDType.toAlgKernel := by
  cases input <;> cases output <;> cases swapped <;> cases causal
  all_goals simp only [kernel,kernel_fp16_fp16,kernel_fp16_bf16,kernel_fp16_fp32,kernel_fp16_fp64,kernel_fp16_f8e4,kernel_fp16_f8e5,kernel_bf16_fp16,kernel_bf16_bf16,kernel_bf16_fp32,kernel_bf16_fp64,kernel_bf16_f8e4,kernel_bf16_f8e5,kernel_fp32_fp16,kernel_fp32_bf16,kernel_fp32_fp32,kernel_fp32_fp64,kernel_fp32_f8e4,kernel_fp32_f8e5,kernel_fp64_fp16,kernel_fp64_bf16,kernel_fp64_fp32,kernel_fp64_fp64,kernel_fp64_f8e4,kernel_fp64_f8e5,kernel_f8e4_fp16,kernel_f8e4_bf16,kernel_f8e4_fp32,kernel_f8e4_fp64,kernel_f8e4_f8e4,kernel_f8e4_f8e5,kernel_f8e5_fp16,kernel_f8e5_bf16,kernel_f8e5_fp32,kernel_f8e5_fp64,kernel_f8e5_f8e4,kernel_f8e5_f8e5,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind,List.map_cons,List.map_nil]
  all_goals simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,List.map_cons,List.map_nil]
  all_goals rfl

/-- Erasure of the unmodified source, including its exact two final stores. -/
theorem actual_body (input : Storage) (Q K V L O : RegionName) (sm_scale : ℝ)
    (stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
      stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
      stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
      BS HEAD SEQLEN BLOCK_M DIM BLOCK_N : Nat) (causal : Bool) (swapped : Bool) :
    (actualKernel input swapped Q K V L O sm_scale stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
    stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
    stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
    stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
    BS HEAD SEQLEN BLOCK_M DIM BLOCK_N causal).eraseDType.toAlgKernel =
    (kernel .fp32 .fp32 swapped Q K V L O sm_scale stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
    stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
    stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
    stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
    BS HEAD SEQLEN BLOCK_M DIM BLOCK_N causal).eraseDType.toAlgKernel := by
  cases input <;> cases swapped <;> cases causal
  all_goals simp only [actualKernel,actual_fp16,actual_bf16,actual_fp32,actual_fp64,actual_f8e4,actual_f8e5,kernel,kernel_fp16_fp16,kernel_fp16_bf16,kernel_fp16_fp32,kernel_fp16_fp64,kernel_fp16_f8e4,kernel_fp16_f8e5,kernel_bf16_fp16,kernel_bf16_bf16,kernel_bf16_fp32,kernel_bf16_fp64,kernel_bf16_f8e4,kernel_bf16_f8e5,kernel_fp32_fp16,kernel_fp32_bf16,kernel_fp32_fp32,kernel_fp32_fp64,kernel_fp32_f8e4,kernel_fp32_f8e5,kernel_fp64_fp16,kernel_fp64_bf16,kernel_fp64_fp32,kernel_fp64_fp64,kernel_fp64_f8e4,kernel_fp64_f8e5,kernel_f8e4_fp16,kernel_f8e4_bf16,kernel_f8e4_fp32,kernel_f8e4_fp64,kernel_f8e4_f8e4,kernel_f8e4_f8e5,kernel_f8e5_fp16,kernel_f8e5_bf16,kernel_f8e5_fp32,kernel_f8e5_fp64,kernel_f8e5_f8e4,kernel_f8e5_f8e5,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind,List.map_cons,List.map_nil]
  all_goals simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,List.map_cons,List.map_nil]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn.Erasure
