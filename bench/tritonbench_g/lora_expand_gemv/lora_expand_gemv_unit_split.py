
import torch
import triton
import triton.language as tl

@triton.jit
def _bgmv_expand_unit_split_kernel(
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
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    CAST_TYPE: tl.constexpr,
):
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


@torch.inference_mode()
def _bgmv_expand(
    inputs: torch.Tensor, lora_b_weights: torch.Tensor, output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor, add_inputs: bool = True,
) -> None:
    tensors = (inputs, lora_b_weights, output_tensor, lora_indices_tensor)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("LoRA GEMV inputs must be tensors")
    if any(t.layout != torch.strided for t in tensors):
        raise ValueError("LoRA GEMV requires strided tensors")
    if (inputs.dtype != torch.float32 or lora_b_weights.dtype != torch.float16 or
            output_tensor.dtype != torch.float32 or lora_indices_tensor.dtype not in (torch.int32, torch.int64)):
        raise TypeError("LoRA GEMV checked profile is FP32 input/output, FP16 weights and signed integer indices")
    if (inputs.ndim != 2 or output_tensor.ndim != 2 or lora_indices_tensor.ndim != 1 or
            lora_b_weights.ndim not in (3, 4)):
        raise ValueError("LoRA GEMV expects matrices, 3D/4D weights and vector indices")
    if lora_b_weights.ndim == 4 and lora_b_weights.shape[1] != 1:
        raise ValueError("four-dimensional weights require a singleton dimension")
    if not all(t.is_cuda and t.device == inputs.device for t in tensors):
        raise ValueError("LoRA GEMV tensors must share a CUDA device")
    if any(not t.is_contiguous() or t.is_neg() or t.is_conj() for t in tensors):
        raise ValueError("LoRA GEMV requires contiguous storage without lazy views")
    if type(add_inputs) is not bool:
        raise TypeError("add_inputs must be a bool")
    B = lora_indices_tensor.numel()
    L, N, K = lora_b_weights.shape[0], lora_b_weights.shape[-2], lora_b_weights.shape[-1]
    C = output_tensor.shape[1]
    if K == 0 or K > 4096:
        raise ValueError("LoRA GEMV rank must be in 1..4096 for the 256-row source tile")
    if inputs.shape[1] != K or B > inputs.shape[0] or B > output_tensor.shape[0] or N > C:
        raise ValueError("LoRA GEMV input, output and index dimensions must fit")
    BLOCK_K = triton.next_power_of_2(K)
    if (B > 65535 or max(L, N, K, C, N * K) >= 2**31 or 2 * N >= 2**31 or
            B * K + BLOCK_K >= 2**31 or L * (N * K) + N * K + 256 * K + BLOCK_K >= 2**31 or
            B * C + N + 256 >= 2**31):
        raise ValueError("LoRA GEMV formed source offsets and launch arguments must fit signed int32")
    output_storage = output_tensor.untyped_storage()
    output_start, output_bytes = output_storage.data_ptr(), output_storage.nbytes()
    for tensor in (inputs, lora_b_weights, lora_indices_tensor):
        read_storage = tensor.untyped_storage()
        read_start, read_bytes = read_storage.data_ptr(), read_storage.nbytes()
        if (output_bytes and read_bytes and
                output_start < read_start + read_bytes and
                read_start < output_start + output_bytes):
            raise ValueError("LoRA GEMV output backing bytes must be separate from read operands")
    if B and N and any(t.numel() and t.data_ptr() % 16 for t in (inputs, lora_b_weights, output_tensor)):
        raise ValueError("nonempty LoRA GEMV data pointers require 16-byte alignment")
    if B == 0:
        return None
    indices = lora_indices_tensor.tolist()
    if any(index != -1 and not 0 <= index < L for index in indices):
        raise ValueError("LoRA GEMV adapter indices must be -1 or name an existing adapter")
    if N == 0:
        return None
    weights = lora_b_weights.squeeze(1) if lora_b_weights.ndim == 4 else lora_b_weights
    with torch.cuda.device(inputs.device):
        _bgmv_expand_unit_split_kernel[(N, B)](
            inputs, weights, output_tensor, N, K, lora_indices_tensor,
            K, 1, N * K, K, 1, C, 1,
            BLOCK_N=256, BLOCK_K=BLOCK_K, SPLIT_N=N,
            EVEN_K=K % BLOCK_K == 0, ADD_INPUTS=add_inputs, CAST_TYPE=True,
        )
    return None
