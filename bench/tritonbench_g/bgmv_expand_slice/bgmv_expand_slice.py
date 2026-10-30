import torch
import triton
import triton.language as tl

@triton.jit
def _bgmv_expand_slice_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    lora_indices,
    xm_stride,
    xk_stride,
    l0_stride,
    lora_k_stride,
    lora_n_stride,
    cm_stride,
    cn_stride,
    slice_offset,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    CAST_TYPE: tl.constexpr,
):
    """
    GroupGEMV, introducing SPLIT_N can improve large hidden_size's performance
    """
    pid_sn = tl.program_id(axis=0).to(tl.int64)
    cur_batch = tl.program_id(axis=1).to(tl.int64)
    lora_index = tl.load(lora_indices + cur_batch).to(tl.int64)
    if lora_index == -1:
        return
    offset_k = tl.arange(0, BLOCK_K).to(tl.int64)
    offset_n = tl.arange(0, BLOCK_N).to(tl.int64)
    if EVEN_K:
        tiled_a = tl.load(input_ptr + cur_batch * xm_stride +
                          offset_k * xk_stride)
    else:
        tiled_a = tl.load(
            input_ptr + cur_batch * xm_stride + offset_k * xk_stride,
            mask=offset_k < K,
            other=0,
        )
    split_n_length = tl.cdiv(tl.cast(N, tl.int64), tl.cast(SPLIT_N, tl.int64))
    if CAST_TYPE:
        tiled_a = tiled_a.to(lora_ptr.dtype.element_ty)
    b_ptr = (lora_ptr + l0_stride * lora_index +
             pid_sn * split_n_length * lora_k_stride)
    c_ptr = (out_ptr + cur_batch * cm_stride + pid_sn * split_n_length +
             slice_offset * cn_stride)

    for n in range(tl.cast(0, tl.int64), split_n_length, BLOCK_N):
        current_n = n + offset_n
        b_ptr_mask = (current_n[:, None] < split_n_length) & (offset_k[None, :]
                                                              < K)
        c_mask = current_n < split_n_length
        tiled_b = tl.load(
            b_ptr + current_n[:, None] * lora_k_stride +
            offset_k[None, :] * lora_n_stride,
            mask=b_ptr_mask,
            other=0.0,
        )
        if ADD_INPUTS:
            tiled_out = tl.load(c_ptr + current_n * cn_stride, mask=c_mask)
            accumulator = tl.sum(tiled_a * tiled_b, 1) + tiled_out
        else:
            accumulator = tl.sum(tiled_a * tiled_b, 1)
        tl.store(c_ptr + current_n * cn_stride, accumulator, mask=c_mask)


@torch.inference_mode()
def _bgmv_expand_slice(inputs: torch.Tensor, lora_b_weights: torch.Tensor,
                       output_tensor: torch.Tensor, lora_indices_tensor: torch.Tensor,
                       slice_offset: int, slice_size: int, add_inputs: bool = True) -> None:
    tensors = (inputs, lora_b_weights, output_tensor, lora_indices_tensor)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("BGMV inputs must be tensors")
    if any(t.layout != torch.strided for t in tensors):
        raise ValueError("BGMV requires strided tensors")
    if inputs.dtype != torch.float32 or lora_b_weights.dtype != torch.float16 or output_tensor.dtype != torch.float32 or lora_indices_tensor.dtype not in (torch.int32, torch.int64):
        raise TypeError("BGMV checked profile is FP32 input/output, FP16 weights, signed integer metadata")
    if inputs.ndim != 2 or output_tensor.ndim != 2 or lora_indices_tensor.ndim != 1 or lora_b_weights.ndim not in (3, 4):
        raise ValueError("BGMV expects X[B,K], W[L,N,K] or W[L,1,N,K], Y[B,C], indices[B]")
    if lora_b_weights.ndim == 4 and lora_b_weights.shape[1] != 1:
        raise ValueError("four-dimensional weights require a singleton dimension")
    if not all(t.is_cuda and t.device == inputs.device for t in tensors):
        raise ValueError("BGMV tensors must share a CUDA device")
    if any(not t.is_contiguous() or t.is_neg() or t.is_conj() for t in tensors):
        raise ValueError("BGMV requires contiguous storage without lazy views")
    if type(slice_offset) is not int or type(slice_size) is not int or type(add_inputs) is not bool:
        raise TypeError("slice coordinates must be Python integers and add_inputs a bool")
    B = lora_indices_tensor.numel()
    L, N, K = lora_b_weights.shape[0], lora_b_weights.shape[-2], lora_b_weights.shape[-1]
    C = output_tensor.shape[1]
    if K == 0 or inputs.shape[1] != K or B > inputs.shape[0] or B > output_tensor.shape[0] or slice_size != N or slice_offset < 0 or slice_offset + N > C:
        raise ValueError("BGMV dimensions and output slice must match")
    if B > 65535 or max(L, N, K, C, N * K) >= 2**31:
        raise ValueError("BGMV grid and scalar strides must fit native launch arguments")
    BLOCK_K = triton.next_power_of_2(K)
    if 4 * (inputs.numel() + BLOCK_K) >= 2**63 or 2 * (lora_b_weights.numel() + 255 * K + BLOCK_K) >= 2**63 or 4 * (output_tensor.numel() + 255) >= 2**63:
        raise ValueError("BGMV formed byte offsets must fit signed 64-bit arithmetic")
    output_storage = output_tensor.untyped_storage()
    output_begin = output_storage.data_ptr()
    output_end = output_begin + output_storage.nbytes()
    for operand in (inputs, lora_b_weights, lora_indices_tensor):
        read_storage = operand.untyped_storage()
        read_begin = read_storage.data_ptr()
        read_end = read_begin + read_storage.nbytes()
        if output_begin < output_end and read_begin < read_end and output_begin < read_end and read_begin < output_end:
            raise ValueError("BGMV output/read backing byte ranges must be disjoint")
    if B and N and any(t.numel() and t.data_ptr() % 16 for t in (inputs, lora_b_weights, output_tensor)):
        raise ValueError("nonempty BGMV data pointers require 16-byte alignment")
    if B == 0 or N == 0:
        return None
    if 256 * BLOCK_K > 2**20:
        raise ValueError("BGMV source tile [256, BLOCK_K] exceeds the static tensor element limit")
    if not bool(torch.all((lora_indices_tensor == -1) | ((lora_indices_tensor >= 0) & (lora_indices_tensor < L))).item()):
        raise ValueError("LoRA metadata must be -1 or an in-range adapter index")
    weights = lora_b_weights.squeeze(1) if lora_b_weights.ndim == 4 else lora_b_weights
    with torch.cuda.device(inputs.device):
        _bgmv_expand_slice_kernel[(N, B)](
            inputs, weights, output_tensor, N, K, lora_indices_tensor,
            K, 1, N * K, K, 1, C, 1, slice_offset,
            BLOCK_N=256, BLOCK_K=BLOCK_K, SPLIT_N=N,
            EVEN_K=K % BLOCK_K == 0, ADD_INPUTS=add_inputs, CAST_TYPE=True,
        )
    return None
