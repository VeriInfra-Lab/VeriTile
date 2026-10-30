
import torch
import triton
import triton.language as tl

@triton.jit
def _triton_rope(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    cos,
    cos_row_stride,
    cos_col_stride,
    sin,
    sin_row_stride,
    sin_col_stride,
    sl,
    bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BACKWARD_PASS: tl.constexpr = False,
):
    pid = tl.program_id(0)

    q_ptr = q_ptr + pid * q_row_stride
    k_ptr = k_ptr + pid * k_row_stride

    cos_row_idx = pid % (sl)
    cos = cos + cos_row_idx * cos_row_stride
    sin = sin + cos_row_idx * sin_row_stride
    cos_offsets = tl.arange(0, pad_hd // 2)
    cos_mask = cos_offsets < hd // 2
    cos_row = tl.load(cos + cos_offsets * cos_col_stride, mask=cos_mask, other=0)
    sin_row = tl.load(sin + cos_offsets * sin_col_stride, mask=cos_mask, other=0)

    first_half_q_offsets = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    first_half_k_offsets = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (tl.arange(0, pad_hd // 2)[None, :] < hd // 2)
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (tl.arange(0, pad_hd // 2)[None, :] < hd // 2)
    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(sin_row.dtype)
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(sin_row.dtype)

    second_half_q_offsets = first_half_q_offsets + (hd // 2)
    second_half_k_offsets = first_half_k_offsets + (hd // 2)
    second_q_mask = first_q_mask
    second_k_mask = first_k_mask
    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=second_q_mask, other=0).to(sin_row.dtype)
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=second_k_mask, other=0).to(sin_row.dtype)

    if not BACKWARD_PASS:
        new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
        tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
        new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
        tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

        new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
        tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
        new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
        tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)
    else:
        new_q_tile_1 = q_tile_1 * cos_row + q_tile_2 * sin_row
        tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
        new_q_tile_2 = q_tile_2 * cos_row - q_tile_1 * sin_row
        tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

        new_k_tile_1 = k_tile_1 * cos_row + k_tile_2 * sin_row
        tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
        new_k_tile_2 = k_tile_2 * cos_row - k_tile_1 * sin_row
        tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)

def _span(t):
    if not t.numel():
        return 0
    return sum((size - 1) * stride for size, stride in zip(t.shape, t.stride())) + 1


def _check_storage(t):
    storage = t.untyped_storage()
    if t.numel() and not (
        storage.data_ptr() <= t.data_ptr()
        and t.data_ptr() + 4 * _span(t) <= storage.data_ptr() + storage.nbytes() <= 2**64
        and t.data_ptr() % 4 == 0
    ):
        raise ValueError("FP32 view must fit aligned backing storage")


def _separate(a, b):
    x, y = a.untyped_storage(), b.untyped_storage()
    if x._cdata == y._cdata or (x.nbytes() and y.nbytes() and not (
        x.data_ptr() + x.nbytes() <= y.data_ptr()
        or y.data_ptr() + y.nbytes() <= x.data_ptr()
    )):
        raise ValueError("gradient and read-only backing allocations must be separate")


def rope_backward(dq, dk, cos, sin):
    """Checked FP32 input adjoint, retaining transpose/contiguous reuse."""
    tensors = (dq, dk, cos, sin)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("expected Tensor arguments")
    if any(t.layout != torch.strided for t in tensors):
        raise ValueError("only strided tensors are supported")
    if any(t.dtype != torch.float32 for t in tensors):
        raise TypeError("the selected profile requires FP32 tensors")
    if not dq.is_cuda or any(t.device != dq.device for t in tensors):
        raise ValueError("all tensors must share one CUDA device")
    if any(t.is_neg() or t.is_conj() for t in tensors):
        raise ValueError("resolved tensor views required")
    if dq.ndim != 4 or dk.ndim != 4 or cos.ndim != 2 or sin.ndim != 2:
        raise ValueError("expected gradients [B,H,S,D] and coefficients [L,D//2]")
    if not dq.is_contiguous() or not dk.is_contiguous():
        raise ValueError("gradient inputs must be contiguous")
    batch_size, n_q_head, seq_len, head_dim = dq.shape
    n_kv_head = dk.shape[1]
    if (dk.shape[0], dk.shape[2], dk.shape[3]) != (batch_size, seq_len, head_dim):
        raise ValueError("gradient batch, sequence and dimension must match")
    if any(t.shape[0] < seq_len or t.shape[1] != head_dim // 2 for t in (cos, sin)):
        raise ValueError("coefficient tables must cover every sequence and half-channel")
    if any(stride < 0 for t in (cos, sin) for stride in t.stride()):
        raise ValueError("nonnegative coefficient strides required")
    for t in tensors:
        _check_storage(t)
    for a, b in ((dq, dk), (dq, cos), (dq, sin), (dk, cos), (dk, sin)):
        _separate(a, b)
    n_row = batch_size * seq_len
    pad_hd = triton.next_power_of_2(max(head_dim, 1))
    pad_n_q_head = triton.next_power_of_2(max(n_q_head, 1))
    pad_n_kv_head = triton.next_power_of_2(max(n_kv_head, 1))
    if max(batch_size, n_row, seq_len, head_dim, n_q_head, n_kv_head,
           n_q_head * head_dim, n_kv_head * head_dim,
           *cos.stride(), *sin.stride()) >= 2**31:
        raise ValueError("launch metadata exceeds signed-i32 profile")
    if pad_hd > 65536 or max(pad_n_q_head, pad_n_kv_head) * pad_hd > 65536:
        raise ValueError("padded tile exceeds checked profile")
    q_last = ((n_row - 1) * n_q_head * head_dim + (pad_n_q_head - 1) * head_dim
              + pad_hd // 2 - 1 + head_dim // 2) if n_row and head_dim >= 2 and (n_q_head or n_kv_head) else 0
    k_last = ((n_row - 1) * n_kv_head * head_dim + (pad_n_kv_head - 1) * head_dim
              + pad_hd // 2 - 1 + head_dim // 2) if n_row and head_dim >= 2 and (n_q_head or n_kv_head) else 0
    c_last = ((seq_len - 1) * cos.stride(0) + (pad_hd // 2 - 1) * cos.stride(1)) if n_row and head_dim >= 2 and (n_q_head or n_kv_head) else 0
    s_last = ((seq_len - 1) * sin.stride(0) + (pad_hd // 2 - 1) * sin.stride(1)) if n_row and head_dim >= 2 and (n_q_head or n_kv_head) else 0
    if max(q_last, k_last, c_last, s_last) >= 2**31:
        raise ValueError("formed padded offsets exceed signed-i32 profile")
    with torch.cuda.device(dq.device):
        dq = dq.transpose(1, 2).contiguous()
        dk = dk.transpose(1, 2).contiguous()
        if n_row == 0 or head_dim < 2 or (n_q_head == 0 and n_kv_head == 0):
            return dq.transpose(1, 2), dk.transpose(1, 2)
        if any(t.data_ptr() + 4 * last + 4 > 2**64
               for t, last in ((dq, q_last), (dk, k_last), (cos, c_last), (sin, s_last))):
            raise ValueError("formed pointers exceed 64-bit profile")
        _triton_rope[(n_row,)](
            dq, n_q_head * head_dim, dk, n_kv_head * head_dim,
            cos, cos.stride(0), cos.stride(1), sin, sin.stride(0), sin.stride(1),
            seq_len, batch_size, n_q_head, n_kv_head, head_dim,
            pad_n_q_head, pad_n_kv_head, pad_hd,
            BLOCK_SIZE=max(pad_n_q_head, pad_n_kv_head), BACKWARD_PASS=True,
        )
        return dq.transpose(1, 2), dk.transpose(1, 2)
