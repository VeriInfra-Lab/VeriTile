
import math
import torch
import triton
import triton.language as tl
# from .utils import get_lora_op_configs

@triton.jit
def _bgmv_shrink_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    lora_indices,
    scaling,
    xm_stride,
    xk_stride,
    l0_stride,
    lora_k_stride,
    lora_n_stride,
    cm_stride,
    cn_stride,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_sk = tl.program_id(axis=0).to(tl.int64)
    cur_batch = tl.program_id(axis=1).to(tl.int64)
    lora_index = tl.load(lora_indices + cur_batch).to(tl.int64)
    if lora_index == -1:
        return

    offset_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offset_k = tl.arange(0, BLOCK_K).to(tl.int64) + pid_sk * BLOCK_K
    a_ptr = input_ptr + cur_batch * xm_stride
    b_ptr = lora_ptr + l0_stride * lora_index
    accumulator = tl.zeros((BLOCK_N, ), dtype=tl.float32)
    for k in range(tl.cast(0, tl.int64), tl.cast(K, tl.int64), BLOCK_K * SPLIT_K):
        current_k = k + offset_k
        current_k_c = tl.max_contiguous(current_k, BLOCK_K)
        tiled_a = tl.load(
            a_ptr + current_k_c,
            mask=current_k < K,
            other=0.0,
        )
        b_ptr_mask = (offset_n[:, None] < N) & (current_k[None, :] < K)

        tiled_b = tl.load(
            b_ptr + offset_n[:, None] * lora_k_stride +
            current_k[None, :] * lora_n_stride,
            mask=b_ptr_mask,
            other=0.0,
        )

        accumulator += tl.sum(tiled_a * tiled_b, 1)
    accumulator *= scaling
    offset_cn = tl.arange(0, BLOCK_N).to(tl.int64)
    c_ptr = out_ptr + cur_batch * cm_stride + offset_cn * cn_stride
    c_mask = offset_cn < N
    if SPLIT_K == 1:
        tl.store(c_ptr, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptr, accumulator, mask=c_mask)


@torch.inference_mode()
def _bgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    scaling: float = 1.0,
) -> None:
    tensors = (inputs, lora_a_weights, output_tensor, lora_indices_tensor)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("BGMV inputs must be tensors")
    if any(t.layout != torch.strided for t in tensors):
        raise ValueError("BGMV requires strided tensors")
    if (inputs.dtype not in (torch.float16, torch.bfloat16) or
            lora_a_weights.dtype != inputs.dtype or output_tensor.dtype != torch.float32 or
            lora_indices_tensor.dtype not in (torch.int32, torch.int64)):
        raise TypeError("BGMV shrink requires matching FP16/BF16 inputs, FP32 output and signed integer metadata")
    if (inputs.ndim != 2 or output_tensor.ndim != 2 or lora_indices_tensor.ndim != 1 or
            lora_a_weights.ndim not in (3, 4)):
        raise ValueError("BGMV expects X[B,K], W[L,N,K] or W[L,1,N,K], Y[B,C], indices[B]")
    if lora_a_weights.ndim == 4 and lora_a_weights.shape[1] != 1:
        raise ValueError("four-dimensional weights require a singleton dimension")
    if not all(t.is_cuda and t.device == inputs.device for t in tensors):
        raise ValueError("BGMV tensors must share a CUDA device")
    if any(not t.is_contiguous() or t.is_neg() or t.is_conj() for t in tensors):
        raise ValueError("BGMV requires contiguous storage without lazy views")
    if type(scaling) is not float or not math.isfinite(scaling):
        raise ValueError("BGMV scaling must be a finite Python float")
    B = lora_indices_tensor.numel()
    L, N, K = lora_a_weights.shape[0], lora_a_weights.shape[-2], lora_a_weights.shape[-1]
    C = output_tensor.shape[1]
    if inputs.shape[1] != K or B > inputs.shape[0] or B > output_tensor.shape[0] or N > C:
        raise ValueError("BGMV dimensions must match and output must contain the active rows and columns")
    if B > 65535 or max(L, N, K, C, N * K) >= 2**31:
        raise ValueError("BGMV grid and scalar strides must fit native launch arguments")
    BLOCK_N = max(1, triton.next_power_of_2(N))
    formed_k = max(16384, triton.cdiv(K, 16384) * 16384)
    if (2 * (inputs.numel() + formed_k) >= 2**63 or
            2 * (lora_a_weights.numel() + BLOCK_N * K + formed_k) >= 2**63 or
            4 * (output_tensor.numel() + BLOCK_N) >= 2**63):
        raise ValueError("BGMV formed byte offsets must fit signed 64-bit arithmetic")
    if B and N and any(t.numel() and t.data_ptr() % 16 for t in (inputs, lora_a_weights, output_tensor)):
        raise ValueError("nonempty BGMV data pointers require 16-byte alignment")
    if B == 0 or N == 0:
        return None
    output_storage = output_tensor.untyped_storage()
    output_start = output_storage.data_ptr()
    output_end = output_start + output_storage.nbytes()
    for tensor in (inputs, lora_a_weights, lora_indices_tensor):
        storage = tensor.untyped_storage()
        start = storage.data_ptr()
        end = start + storage.nbytes()
        if output_start < output_end and start < end and output_start < end and start < output_end:
            raise ValueError("BGMV output backing byte range must be separate from its read operands")
    if 256 * BLOCK_N > 2**20:
        raise ValueError("BGMV broadcast tile must contain at most 2**20 elements")
    if not bool(torch.all((lora_indices_tensor == -1) |
                          ((lora_indices_tensor >= 0) & (lora_indices_tensor < L))).item()):
        raise ValueError("LoRA metadata must be -1 or an in-range adapter index")
    weights = lora_a_weights.squeeze(1) if lora_a_weights.ndim == 4 else lora_a_weights
    with torch.cuda.device(inputs.device):
        _bgmv_shrink_kernel[(64, B)](
            inputs, weights, output_tensor, N, K, lora_indices_tensor, scaling,
            K, 1, N * K, K, 1, C, 1,
            BLOCK_N=BLOCK_N, BLOCK_K=256, SPLIT_K=64,
        )
    return None
