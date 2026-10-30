"""Separate SGMV repair; original SHA256: a3643842b99a76379b1a8c8be29b2f531186c776ddc3d0d1583edaccc7e1a281."""
import torch
import triton
import triton.language as tl

@triton.jit
def _sgmv_expand_slice_kernel_repaired(
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    b_seq_start_loc,
    seq_lens,
    lora_indices,
    xm_stride,
    xk_stride,  # 1
    l0_stride,  # hidden_size*max_rank
    lora_k_stride,
    lora_n_stride,
    cm_stride,
    cn_stride,
    slice_offset,
    batch,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    CAST_TYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    cur_batch = tl.cast(batch, tl.int64)
    N = tl.cast(N, tl.int64)
    K = tl.cast(K, tl.int64)
    xm_stride = tl.cast(xm_stride, tl.int64)
    xk_stride = tl.cast(xk_stride, tl.int64)
    l0_stride = tl.cast(l0_stride, tl.int64)
    lora_k_stride = tl.cast(lora_k_stride, tl.int64)
    lora_n_stride = tl.cast(lora_n_stride, tl.int64)
    cm_stride = tl.cast(cm_stride, tl.int64)
    cn_stride = tl.cast(cn_stride, tl.int64)
    slice_offset = tl.cast(slice_offset, tl.int64)
    cta_n_num = (N + BLOCK_N - 1) // BLOCK_N
    pid_m = pid // cta_n_num
    pid_n = pid % cta_n_num
    M = tl.load(seq_lens + cur_batch).to(tl.int64)
    if M <= 0:
        return
    if pid_m * BLOCK_M >= M:
        return
    lora_index = tl.load(lora_indices + cur_batch).to(tl.int64)
    if lora_index == -1:
        return
    cur_seq_start = tl.load(b_seq_start_loc + cur_batch).to(tl.int64)
    offset_m = tl.arange(0, BLOCK_M).to(tl.int64) + pid_m * BLOCK_M
    offset_n = tl.arange(0, BLOCK_N).to(tl.int64) + pid_n * BLOCK_N
    offset_k = tl.arange(0, BLOCK_K).to(tl.int64)
    ram = offset_m % M
    rbn = offset_n % N

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range((K + BLOCK_K - 1) // BLOCK_K):
        global_k = k * BLOCK_K + offset_k
        safe_k = tl.where(global_k < K, global_k, 0)
        a_ptr = (input_ptr + cur_seq_start * xm_stride + ram[:, None] * xm_stride +
                 safe_k[None, :] * xk_stride)
        b_ptr = (lora_ptr + l0_stride * lora_index +
                 safe_k[:, None] * lora_n_stride + rbn[None, :] * lora_k_stride)
        if EVEN_K:
            tiled_a = tl.load(a_ptr)
            tiled_b = tl.load(b_ptr)
        else:
            tiled_a = tl.load(a_ptr,
                              mask=global_k[None, :] < K,
                              other=0)
            tiled_b = tl.load(b_ptr,
                              mask=global_k[:, None] < K,
                              other=0)
        if CAST_TYPE:
            tiled_a = tiled_a.to(lora_ptr.dtype.element_ty)
        accumulator = tl.dot(tiled_a, tiled_b, accumulator)
    tiled_c = accumulator.to(lora_ptr.dtype.element_ty)
    offset_cm = cur_seq_start + tl.arange(0, BLOCK_M).to(tl.int64) + pid_m * BLOCK_M
    offset_cn = tl.arange(0, BLOCK_N).to(tl.int64) + pid_n * BLOCK_N + slice_offset
    safe_cm = tl.where(offset_cm < cur_seq_start + M, offset_cm, 0)
    safe_cn = tl.where(offset_cn < slice_offset + N, offset_cn, slice_offset)
    c_ptr = (out_ptr + safe_cm[:, None] * cm_stride +
             safe_cn[None, :] * cn_stride)
    M = tl.load(seq_lens + cur_batch).to(tl.int64)
    c_mask = (offset_cm[:, None] < cur_seq_start + M) & (offset_cn[None, :] < slice_offset + N)
    if ADD_INPUTS:
        tiled_out = tl.load(c_ptr, mask=c_mask)
        tiled_c += tiled_out
    tl.store(c_ptr, tiled_c, mask=c_mask)


@torch.inference_mode()
def _sgmv_expand_slice_repaired(
    inputs: torch.Tensor, lora_b_weights: torch.Tensor, output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor, seq_len_tensor: torch.Tensor, lora_indices_tensor: torch.Tensor,
    batches: int, max_seq_length: int, token_nums: int, slice_offset: int, slice_size: int,
    add_inputs: bool = False,
) -> None:
    metadata = (b_seq_start_loc, seq_len_tensor, lora_indices_tensor)
    tensors = (inputs, lora_b_weights, output_tensor) + metadata
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("SGMV arguments must be tensors")
    if any(t.layout != torch.strided for t in tensors):
        raise ValueError("SGMV requires strided tensors")
    if (inputs.dtype != torch.float32 or lora_b_weights.dtype != torch.float16 or
            output_tensor.dtype != torch.float32 or any(t.dtype not in (torch.int32, torch.int64) for t in metadata)):
        raise TypeError("SGMV checked profile is FP32 input/output, FP16 weights and signed integer metadata")
    if (inputs.ndim != 2 or output_tensor.ndim != 2 or lora_b_weights.ndim not in (3, 4) or
            any(t.ndim != 1 for t in metadata)):
        raise ValueError("SGMV requires input/output matrices, 3D/4D weights and vector metadata")
    if lora_b_weights.ndim == 4 and lora_b_weights.shape[1] != 1:
        raise ValueError("four-dimensional weights require a singleton dimension")
    if not all(t.is_cuda and t.device == inputs.device for t in tensors):
        raise ValueError("SGMV tensors must share a CUDA device")
    if any(not t.is_contiguous() or t.is_neg() or t.is_conj() for t in tensors):
        raise ValueError("SGMV requires contiguous storage without lazy views")
    if any(type(v) is not int or v < 0 for v in
           (batches, max_seq_length, token_nums, slice_offset, slice_size)) or type(add_inputs) is not bool:
        raise ValueError("SGMV sizes must be nonnegative Python integers and add_inputs a bool")
    L, N, K = lora_b_weights.shape[0], lora_b_weights.shape[-2], lora_b_weights.shape[-1]
    C = output_tensor.shape[1]
    if (inputs.shape != (token_nums, K) or slice_size != N or slice_offset + N > C or
            b_seq_start_loc.numel() != batches or lora_indices_tensor.numel() != batches or seq_len_tensor.numel() < batches):
        raise ValueError("SGMV shapes, output slice and metadata lengths must match")
    if max(batches, token_nums, output_tensor.shape[0], L, N, K, C, N * K, slice_offset) >= 2**31:
        raise ValueError("SGMV scalar strides and coordinates must fit native launch arguments")
    if (4 * inputs.numel() >= 2**63 or 2 * lora_b_weights.numel() >= 2**63 or
            4 * output_tensor.numel() >= 2**63 or any(t.numel() * t.element_size() >= 2**63 for t in metadata)):
        raise ValueError("SGMV byte extents must fit signed 64-bit arithmetic")
    output_storage = output_tensor.untyped_storage()
    output_start, output_bytes = output_storage.data_ptr(), output_storage.nbytes()
    for tensor in (inputs, lora_b_weights) + metadata:
        read_storage = tensor.untyped_storage()
        read_start, read_bytes = read_storage.data_ptr(), read_storage.nbytes()
        if (output_bytes and read_bytes and
                output_start < read_start + read_bytes and
                read_start < output_start + output_bytes):
            raise ValueError("SGMV output backing bytes must be separate from read operands")
    if batches and N and any(t.numel() and t.data_ptr() % 16 for t in (inputs, lora_b_weights, output_tensor)):
        raise ValueError("nonempty SGMV data pointers require 16-byte alignment")
    if batches == 0:
        return None
    # Host metadata inspection is required for validation and serial scheduling;
    # the kernel still receives the original metadata buffers.
    start_values = b_seq_start_loc.tolist()
    length_values = seq_len_tensor.tolist()[:batches]
    index_values = lora_indices_tensor.tolist()
    for start, length, adapter in zip(start_values, length_values, index_values):
        if length < 0:
            raise ValueError("segment length must be nonnegative")
        if length and adapter != -1:
            if start < 0 or start + length > min(token_nums, output_tensor.shape[0]) or not 0 <= adapter < L:
                raise ValueError("active segment or adapter is outside its allocation")
            if triton.cdiv(length, 32) * triton.cdiv(N, 32) > 2**31 - 1:
                raise ValueError("SGMV segment grid exceeds the CUDA x-axis limit")
    if N == 0 or not any(length and adapter != -1 for length, adapter in zip(length_values, index_values)):
        return None
    weights = lora_b_weights.squeeze(1) if lora_b_weights.ndim == 4 else lora_b_weights
    with torch.cuda.device(inputs.device):
        for batch, (length, adapter) in enumerate(zip(length_values, index_values)):
            if length == 0 or adapter == -1:
                continue
            grid = (triton.cdiv(length, 32) * triton.cdiv(N, 32),)
            _sgmv_expand_slice_kernel_repaired[grid](
                inputs, weights, output_tensor, N, K, b_seq_start_loc, seq_len_tensor, lora_indices_tensor,
                K, 1, N * K, K, 1, C, 1, slice_offset, batch,
                BLOCK_M=32, BLOCK_N=32, BLOCK_K=16,
                EVEN_K=K % 16 == 0, ADD_INPUTS=add_inputs, CAST_TYPE=True, num_warps=4, num_ctas=1,
            )
    return None


@triton.jit
def _sgmv_expand_slice_kernel_index_reuse(
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    b_seq_start_loc,
    seq_lens,
    lora_indices,
    xm_stride,
    xk_stride,  # 1
    l0_stride,  # hidden_size*max_rank
    lora_k_stride,
    lora_n_stride,
    cm_stride,
    cn_stride,
    slice_offset,
    batch,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    CAST_TYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    cur_batch = tl.cast(batch, tl.int64)
    N = tl.cast(N, tl.int64)
    K = tl.cast(K, tl.int64)
    xm_stride = tl.cast(xm_stride, tl.int64)
    xk_stride = tl.cast(xk_stride, tl.int64)
    l0_stride = tl.cast(l0_stride, tl.int64)
    lora_k_stride = tl.cast(lora_k_stride, tl.int64)
    lora_n_stride = tl.cast(lora_n_stride, tl.int64)
    cm_stride = tl.cast(cm_stride, tl.int64)
    cn_stride = tl.cast(cn_stride, tl.int64)
    slice_offset = tl.cast(slice_offset, tl.int64)
    cta_n_num = (N + BLOCK_N - 1) // BLOCK_N
    pid_m = pid // cta_n_num
    pid_n = pid % cta_n_num
    M = tl.load(seq_lens + cur_batch).to(tl.int64)
    if M <= 0:
        return
    if pid_m * BLOCK_M >= M:
        return
    lora_index = tl.load(lora_indices + cur_batch).to(tl.int64)
    if lora_index == -1:
        return
    cur_seq_start = tl.load(b_seq_start_loc + cur_batch).to(tl.int64)
    offset_m = tl.arange(0, BLOCK_M).to(tl.int64) + pid_m * BLOCK_M
    offset_n = tl.arange(0, BLOCK_N).to(tl.int64) + pid_n * BLOCK_N
    offset_k = tl.arange(0, BLOCK_K).to(tl.int64)
    ram = offset_m % M
    rbn = offset_n % N

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range((K + BLOCK_K - 1) // BLOCK_K):
        global_k = k * BLOCK_K + offset_k
        safe_k = tl.where(global_k < K, global_k, 0)
        a_ptr = (input_ptr + cur_seq_start * xm_stride + ram[:, None] * xm_stride +
                 safe_k[None, :] * xk_stride)
        b_ptr = (lora_ptr + l0_stride * lora_index +
                 safe_k[:, None] * lora_n_stride + rbn[None, :] * lora_k_stride)
        if EVEN_K:
            tiled_a = tl.load(a_ptr)
            tiled_b = tl.load(b_ptr)
        else:
            tiled_a = tl.load(a_ptr,
                              mask=global_k[None, :] < K,
                              other=0)
            tiled_b = tl.load(b_ptr,
                              mask=global_k[:, None] < K,
                              other=0)
        if CAST_TYPE:
            tiled_a = tiled_a.to(lora_ptr.dtype.element_ty)
        accumulator = tl.dot(tiled_a, tiled_b, accumulator)
    tiled_c = accumulator.to(lora_ptr.dtype.element_ty)
    offset_cm = cur_seq_start + offset_m
    offset_cn = offset_n + slice_offset
    safe_cm = tl.where(offset_cm < cur_seq_start + M, offset_cm, 0)
    safe_cn = tl.where(offset_cn < slice_offset + N, offset_cn, slice_offset)
    c_ptr = (out_ptr + safe_cm[:, None] * cm_stride +
             safe_cn[None, :] * cn_stride)
    M = tl.load(seq_lens + cur_batch).to(tl.int64)
    c_mask = (offset_cm[:, None] < cur_seq_start + M) & (offset_cn[None, :] < slice_offset + N)
    if ADD_INPUTS:
        tiled_out = tl.load(c_ptr, mask=c_mask)
        tiled_c += tiled_out
    tl.store(c_ptr, tiled_c, mask=c_mask)




@torch.inference_mode()
def _sgmv_expand_slice_index_reuse(
    inputs: torch.Tensor, lora_b_weights: torch.Tensor, output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor, seq_len_tensor: torch.Tensor, lora_indices_tensor: torch.Tensor,
    batches: int, max_seq_length: int, token_nums: int, slice_offset: int, slice_size: int,
    add_inputs: bool = False,
) -> None:
    metadata = (b_seq_start_loc, seq_len_tensor, lora_indices_tensor)
    tensors = (inputs, lora_b_weights, output_tensor) + metadata
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("SGMV arguments must be tensors")
    if any(t.layout != torch.strided for t in tensors):
        raise ValueError("SGMV requires strided tensors")
    if (inputs.dtype != torch.float32 or lora_b_weights.dtype != torch.float16 or
            output_tensor.dtype != torch.float32 or any(t.dtype not in (torch.int32, torch.int64) for t in metadata)):
        raise TypeError("SGMV checked profile is FP32 input/output, FP16 weights and signed integer metadata")
    if (inputs.ndim != 2 or output_tensor.ndim != 2 or lora_b_weights.ndim not in (3, 4) or
            any(t.ndim != 1 for t in metadata)):
        raise ValueError("SGMV requires input/output matrices, 3D/4D weights and vector metadata")
    if lora_b_weights.ndim == 4 and lora_b_weights.shape[1] != 1:
        raise ValueError("four-dimensional weights require a singleton dimension")
    if not all(t.is_cuda and t.device == inputs.device for t in tensors):
        raise ValueError("SGMV tensors must share a CUDA device")
    if any(not t.is_contiguous() or t.is_neg() or t.is_conj() for t in tensors):
        raise ValueError("SGMV requires contiguous storage without lazy views")
    if any(type(v) is not int or v < 0 for v in
           (batches, max_seq_length, token_nums, slice_offset, slice_size)) or type(add_inputs) is not bool:
        raise ValueError("SGMV sizes must be nonnegative Python integers and add_inputs a bool")
    L, N, K = lora_b_weights.shape[0], lora_b_weights.shape[-2], lora_b_weights.shape[-1]
    C = output_tensor.shape[1]
    if (inputs.shape != (token_nums, K) or slice_size != N or slice_offset + N > C or
            b_seq_start_loc.numel() != batches or lora_indices_tensor.numel() != batches or seq_len_tensor.numel() < batches):
        raise ValueError("SGMV shapes, output slice and metadata lengths must match")
    if max(batches, token_nums, output_tensor.shape[0], L, N, K, C, N * K, slice_offset) >= 2**31:
        raise ValueError("SGMV scalar strides and coordinates must fit native launch arguments")
    if (4 * inputs.numel() >= 2**63 or 2 * lora_b_weights.numel() >= 2**63 or
            4 * output_tensor.numel() >= 2**63 or any(t.numel() * t.element_size() >= 2**63 for t in metadata)):
        raise ValueError("SGMV byte extents must fit signed 64-bit arithmetic")
    output_storage = output_tensor.untyped_storage()
    output_start, output_bytes = output_storage.data_ptr(), output_storage.nbytes()
    for tensor in (inputs, lora_b_weights) + metadata:
        read_storage = tensor.untyped_storage()
        read_start, read_bytes = read_storage.data_ptr(), read_storage.nbytes()
        if (output_bytes and read_bytes and
                output_start < read_start + read_bytes and
                read_start < output_start + output_bytes):
            raise ValueError("SGMV output backing bytes must be separate from read operands")
    if batches and N and any(t.numel() and t.data_ptr() % 16 for t in (inputs, lora_b_weights, output_tensor)):
        raise ValueError("nonempty SGMV data pointers require 16-byte alignment")
    if batches == 0:
        return None
    # Host metadata inspection is required for validation and serial scheduling;
    # the kernel still receives the original metadata buffers.
    start_values = b_seq_start_loc.tolist()
    length_values = seq_len_tensor.tolist()[:batches]
    index_values = lora_indices_tensor.tolist()
    for start, length, adapter in zip(start_values, length_values, index_values):
        if length < 0:
            raise ValueError("segment length must be nonnegative")
        if length and adapter != -1:
            if start < 0 or start + length > min(token_nums, output_tensor.shape[0]) or not 0 <= adapter < L:
                raise ValueError("active segment or adapter is outside its allocation")
            if triton.cdiv(length, 32) * triton.cdiv(N, 32) > 2**31 - 1:
                raise ValueError("SGMV segment grid exceeds the CUDA x-axis limit")
    if N == 0 or not any(length and adapter != -1 for length, adapter in zip(length_values, index_values)):
        return None
    weights = lora_b_weights.squeeze(1) if lora_b_weights.ndim == 4 else lora_b_weights
    with torch.cuda.device(inputs.device):
        for batch, (length, adapter) in enumerate(zip(length_values, index_values)):
            if length == 0 or adapter == -1:
                continue
            grid = (triton.cdiv(length, 32) * triton.cdiv(N, 32),)
            _sgmv_expand_slice_kernel_index_reuse[grid](
                inputs, weights, output_tensor, N, K, b_seq_start_loc, seq_len_tensor, lora_indices_tensor,
                K, 1, N * K, K, 1, C, 1, slice_offset, batch,
                BLOCK_M=32, BLOCK_N=32, BLOCK_K=16,
                EVEN_K=K % 16 == 0, ADD_INPUTS=add_inputs, CAST_TYPE=True, num_warps=4, num_ctas=1,
            )
    return None
