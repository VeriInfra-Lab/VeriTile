
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
    sin,
    sin_row_stride,
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
    cos_row = tl.load(cos + cos_offsets, mask=cos_mask, other=0)
    sin_row = tl.load(sin + cos_offsets, mask=cos_mask, other=0)

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



def rope_forward(q, k, cos, sin):
    """Checked FP32 forward RoPE, preserving the original four return views.

    Q/K are contiguous [B,H,S,D], coefficients are contiguous [S,D/2].
    Transpose+contiguous may reuse Q/K storage when a transposed axis is
    singleton; those inputs are intentionally updated in place in that case.
    """
    tensors = (q, k, cos, sin)
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("Q, K, cos and sin must be tensors")
    if any(t.layout != torch.strided for t in tensors):
        raise ValueError("only strided tensors are supported")
    if any(t.dtype != torch.float32 for t in tensors):
        raise TypeError("Q, K, cos and sin must all be FP32")
    if q.ndim != 4 or k.ndim != 4 or cos.ndim != 2 or sin.ndim != 2:
        raise ValueError("expected Q/K rank 4 and cos/sin rank 2")
    if any(not t.is_cuda for t in tensors) or any(t.device != q.device for t in tensors):
        raise ValueError("all tensors must be on the same CUDA device")
    if any(not t.is_contiguous() or t.is_neg() or t.is_conj() for t in tensors):
        raise ValueError("only contiguous tensors without lazy metadata are supported")
    batch_size, n_q_head, seq_len, head_dim = q.shape
    n_kv_head = k.shape[1]
    if n_q_head <= 0 or n_kv_head <= 0 or head_dim < 2 or head_dim % 2:
        raise ValueError("head counts must be positive and head_dim positive and even")
    if k.shape != (batch_size, n_kv_head, seq_len, head_dim):
        raise ValueError("Q and K must agree on batch, sequence and head dimensions")
    if cos.shape != (seq_len, head_dim // 2) or sin.shape != cos.shape:
        raise ValueError("coefficient shape must be exactly [S,D/2]")
    # Four independent backing allocations, including any guard/prefix cells.
    stores = [t.untyped_storage() for t in tensors]
    for i, a in enumerate(stores):
        for b in stores[i + 1:]:
            if a.nbytes() and b.nbytes():
                if not (a.data_ptr() + a.nbytes() <= b.data_ptr()
                        or b.data_ptr() + b.nbytes() <= a.data_ptr()):
                    raise ValueError("the four tensors must have separate backing allocations")
    for t, storage in zip(tensors, stores):
        if t.numel() and not (storage.data_ptr() <= t.data_ptr()
                and t.data_ptr() + 4 * t.numel() <= storage.data_ptr() + storage.nbytes()
                and storage.data_ptr() + storage.nbytes() <= 2**64):
            raise ValueError("tensor view must fit its backing storage")
        if t.numel() and t.data_ptr() % 4:
            raise ValueError("FP32 data pointers must be four-byte aligned")
    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)
    BLOCK_SIZE = max(pad_n_q_head, pad_n_kv_head)
    n_row = batch_size * seq_len
    q_row_stride = n_q_head * head_dim
    k_row_stride = n_kv_head * head_dim
    if max(pad_n_q_head, pad_n_kv_head) * (pad_hd // 2) > 1048576:
        raise ValueError("Triton tile exceeds the supported element limit")
    if max(batch_size, seq_len, n_q_head, n_kv_head, head_dim,
           pad_hd, pad_n_q_head, pad_n_kv_head, n_row,
           q_row_stride, k_row_stride) >= 2**31:
        raise ValueError("launch dimensions exceed the signed-i32 profile")
    if n_row:
        q_last = ((n_row - 1) * q_row_stride + (pad_n_q_head - 1) * head_dim
                  + pad_hd // 2 - 1 + head_dim // 2)
        k_last = ((n_row - 1) * k_row_stride + (pad_n_kv_head - 1) * head_dim
                  + pad_hd // 2 - 1 + head_dim // 2)
        coefficient_last = (seq_len - 1) * (head_dim // 2) + pad_hd // 2 - 1
        if max(q_last, k_last, coefficient_last) >= 2**31:
            raise ValueError("formed padded addresses exceed the signed-i32 profile")
    with torch.cuda.device(q.device):
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        cos = cos.contiguous()
        sin = sin.contiguous()
        if n_row:
            if any(t.data_ptr() + 4 * last + 4 > 2**64 for t, last in
                   ((q, q_last), (k, k_last), (cos, coefficient_last), (sin, coefficient_last))):
                raise ValueError("formed pointers exceed the 64-bit address profile")
            _triton_rope[(n_row,)](
                q, q_row_stride, k, k_row_stride,
                cos, head_dim // 2, sin, head_dim // 2,
                seq_len, batch_size, n_q_head, n_kv_head, head_dim,
                pad_n_q_head, pad_n_kv_head, pad_hd,
                BLOCK_SIZE=BLOCK_SIZE, BACKWARD_PASS=False,
            )
        return q.transpose(1, 2), k.transpose(1, 2), cos, sin
