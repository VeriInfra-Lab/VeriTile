import bench.tritonbench_g.lora_expand_gemv.Proofs.Support.SourceTypes
import VeriTile.Triton.DSL.PythonFullConstructors
namespace VeriTile.Bench.Repairs.LoraExpandGemv.Pasted
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LoraExpandGemv.FPValue (OutputFormat)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 40000
inductive IndexStorage where | i32 | i64 deriving DecidableEq, Repr
def callSignature (input_format weight_format output_format : OutputFormat) (index_storage : IndexStorage) : CallSignature where
  pointers := [("input_ptr", match input_format with | .half => .fp16 | .bfloat => .bf16 | .single => .fp32),
    ("lora_ptr", match weight_format with | .half => .fp16 | .bfloat => .bf16 | .single => .fp32),
    ("out_ptr", match output_format with | .half => .fp16 | .bfloat => .bf16 | .single => .fp32),
    ("lora_indices",match index_storage with | .i32 => .i32 | .i64 => .i64)]
  scalars := [("N",.integer),("K",.integer),("xm_stride",.integer),("xk_stride",.integer),("l0_stride",.integer),("lora_k_stride",.integer),("lora_n_stride",.integer),("cm_stride",.integer),("cn_stride",.integer)]
  constexpr := ["BLOCK_N","BLOCK_K","SPLIT_N","EVEN_K","ADD_INPUTS","CAST_TYPE"]
  readPorts := ["input_ptr","lora_ptr","out_ptr","lora_indices"]
  writePorts := ["out_ptr"]
def originalKernel (input_format weight_format output_format : OutputFormat) (index_storage : IndexStorage)
    (input_ptr lora_ptr out_ptr lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride : Nat) (BLOCK_N BLOCK_K SPLIT_N : Nat) (EVEN_K ADD_INPUTS CAST_TYPE : Bool) : ComputeKernel :=
  with_triton_call (callSignature input_format weight_format output_format index_storage)
    call_args [N := N,K := K,xm_stride := xm_stride,xk_stride := xk_stride,l0_stride := l0_stride,lora_k_stride := lora_k_stride,lora_n_stride := lora_n_stride,cm_stride := cm_stride,cn_stride := cn_stride] in triton {
    """
    GroupGEMV, additionally, introducing SPLIT_N can improve large hidden_size's
    performance
    """
    pid_sn = tl.program_id(axis=0)
    cur_batch = tl.program_id(axis=1)
    lora_index = tl.load(lora_indices + cur_batch)
    if lora_index == -1:
        return
    offset_k = tl.arange(0, BLOCK_K)
    offset_n = tl.arange(0, BLOCK_N)
    if EVEN_K:
        tiled_a = tl.load(input_ptr + cur_batch * xm_stride +
                          offset_k * xk_stride, )  # [BLOCK_K]
    else:
        tiled_a = tl.load(
            input_ptr + cur_batch * xm_stride + offset_k * xk_stride,
            mask=offset_k < K,
            other=0,
        )  # [BLOCK_K]
    # N must be divisible by SPLIT_N
    split_n_length = tl.cdiv(N, SPLIT_N)
    if CAST_TYPE:
        tiled_a = tiled_a.to(lora_ptr.dtype.element_ty)
    # sliding  to  next row-block
    b_ptr = (lora_ptr + l0_stride * lora_index +
             pid_sn * split_n_length * lora_k_stride)
    c_ptr = out_ptr + cur_batch * cm_stride + pid_sn * split_n_length
    for n in range(0, split_n_length, BLOCK_N):
        current_n = n + offset_n
        current_n_c = tl.max_contiguous(current_n, BLOCK_N)
        b_ptr_mask = (current_n[:, None] < split_n_length) & (offset_k[None, :]
                                                              < K)
        c_mask = current_n < split_n_length
        tiled_b = tl.load(
            b_ptr + current_n_c[:, None] * lora_k_stride +
            offset_k[None, :] * lora_n_stride,
            mask=b_ptr_mask,
            other=0.0,
        )  # [BLOCK_N,BLOCK_K]
        if ADD_INPUTS:
            tiled_out = tl.load(c_ptr + current_n * cn_stride, mask=c_mask)
            accumulator = tl.sum(tiled_a * tiled_b, 1) + tiled_out
        else:
            accumulator = tl.sum(tiled_a * tiled_b, 1)

        tl.store(c_ptr + current_n * cn_stride, accumulator, mask=c_mask)
  }


def candidateKernel (input_format weight_format output_format : OutputFormat) (index_storage : IndexStorage)
    (input_ptr lora_ptr out_ptr lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride : Nat) (BLOCK_N BLOCK_K SPLIT_N : Nat) (EVEN_K ADD_INPUTS CAST_TYPE : Bool) : ComputeKernel :=
  with_triton_call (callSignature input_format weight_format output_format index_storage)
    call_args [N := N,K := K,xm_stride := xm_stride,xk_stride := xk_stride,l0_stride := l0_stride,lora_k_stride := lora_k_stride,lora_n_stride := lora_n_stride,cm_stride := cm_stride,cn_stride := cn_stride] in triton {
    """
    GroupGEMV, additionally, introducing SPLIT_N can improve large hidden_size's
    performance
    """
    pid_sn = tl.program_id(axis=0)
    cur_batch = tl.program_id(axis=1)
    lora_index = tl.load(lora_indices + cur_batch)
    if lora_index == -1:
        return
    offset_k = tl.arange(0, BLOCK_K)
    offset_n = tl.arange(0, BLOCK_N)
    if EVEN_K:
        tiled_a = tl.load(input_ptr + cur_batch * xm_stride +
                          offset_k * xk_stride, )  # [BLOCK_K]
    else:
        tiled_a = tl.load(
            input_ptr + cur_batch * xm_stride + offset_k * xk_stride,
            mask=offset_k < K,
            other=0,
        )  # [BLOCK_K]
    # The checked wrapper passes SPLIT_N=N and returns before N=0.
    split_n_length = 1
    if CAST_TYPE:
        tiled_a = tiled_a.to(lora_ptr.dtype.element_ty)
    # sliding  to  next row-block
    b_ptr = (lora_ptr + l0_stride * lora_index +
             pid_sn * split_n_length * lora_k_stride)
    c_ptr = out_ptr + cur_batch * cm_stride + pid_sn * split_n_length
    for n in range(0, split_n_length, BLOCK_N):
        current_n = n + offset_n
        current_n_c = tl.max_contiguous(current_n, BLOCK_N)
        b_ptr_mask = (current_n[:, None] < split_n_length) & (offset_k[None, :]
                                                              < K)
        c_mask = current_n < split_n_length
        tiled_b = tl.load(
            b_ptr + current_n_c[:, None] * lora_k_stride +
            offset_k[None, :] * lora_n_stride,
            mask=b_ptr_mask,
            other=0.0,
        )  # [BLOCK_N,BLOCK_K]
        if ADD_INPUTS:
            tiled_out = tl.load(c_ptr + current_n * cn_stride, mask=c_mask)
            accumulator = tl.sum(tiled_a * tiled_b, 1) + tiled_out
        else:
            accumulator = tl.sum(tiled_a * tiled_b, 1)

        tl.store(c_ptr + current_n * cn_stride, accumulator, mask=c_mask)
  }

end
end VeriTile.Bench.Repairs.LoraExpandGemv.Pasted
namespace VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches
open VeriTile.Triton
open FPValue (OutputFormat)
noncomputable section
/-- Every retained invocation profile uses the single pasted repaired body. -/
def source (profile : InputProfile) (out : OutputFormat) (p : Parameters) : ComputeKernel :=
  VeriTile.Bench.Repairs.LoraExpandGemv.Pasted.originalKernel
    (inputFormat profile) (weightFormat profile) out .i64 p.input p.weight p.output p.indices.cast
    p.outputSize p.inner p.inputBatchStride p.inputInnerStride p.loraStride
    p.weightRowStride p.weightInnerStride p.outputBatchStride p.outputStride
    p.blockRows p.blockInner p.splits p.evenK p.addInputs p.castType
end
end VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches
namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LoraExpandGemv
noncomputable section
def fullSource (p : Branches.Parameters) : ComputeKernel :=
  Branches.source .singleHalf .single p

def source (profile : Branches.InputProfile) (out : FPValue.OutputFormat)
    (p : Branches.Parameters) : ComputeKernel :=
  Branches.source profile out p
end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated
