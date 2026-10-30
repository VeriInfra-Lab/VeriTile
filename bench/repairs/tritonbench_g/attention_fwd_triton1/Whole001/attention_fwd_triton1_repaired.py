"""FP32 chunked linear attention with explicit dense-input and launch bounds."""
import torch
import triton
import triton.language as tl


@triton.jit
def attention_fwd_kernel(
    q, k, v, h, o, s_qh, s_qt, s_qd, s_hh, s_ht, T, scale,
    BT: tl.constexpr, BD: tl.constexpr, NT: tl.constexpr,
    STORE: tl.constexpr, IFCOND: tl.constexpr,
):
    i_bh = tl.program_id(0).to(tl.int64)
    b_h = tl.zeros([BD, BD], dtype=tl.float32)
    for i in range(0, NT):
        # make_block_ptr offsets are signed32. The host checks both padded
        # temporal coordinates and every pre-update H snapshot coordinate.
        p_q = tl.make_block_ptr(q + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i * BT, 0), (BT, BD), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qh, (BD, T), (s_qd, s_qt), (0, i * BT), (BD, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i * BT, 0), (BT, BD), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_hh, (NT * BD, BD), (s_ht, 1), (i * BD, 0), (BD, BD), (1, 0))
        p_o = tl.make_block_ptr(o + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i * BT, 0), (BT, BD), (1, 0))
        if STORE:
            tl.store(p_h, b_h.to(p_h.dtype.element_ty))
        b_q = tl.load(p_q, boundary_check=(0,), padding_option="zero")
        b_q = (b_q * scale).to(b_q.dtype)
        b_k = tl.load(p_k, boundary_check=(1,), padding_option="zero")
        b_v = tl.load(p_v, boundary_check=(0,), padding_option="zero")
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_o = tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)
        if IFCOND:
            if i == 0:
                b_h = tl.dot(b_k, b_v, allow_tf32=False)
            else:
                b_o += tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False)
                b_h += tl.dot(b_k, b_v, allow_tf32=False)
        else:
            b_o += tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False)
            b_h += tl.dot(b_k, b_v, allow_tf32=False)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


def _launch_domain(batch, heads, length, dim):
    """Bounds on every source coordinate, including inactive tail lanes."""
    chunks = triton.cdiv(length, 32)
    padded = chunks * 32
    head_stride = length * dim
    history_rows = chunks * dim
    history_head_stride = history_rows * dim
    values = (batch, heads, length, dim, chunks, padded, head_stride,
              history_rows, history_head_stride, batch * heads)
    if not all(0 <= value <= (1 << 31) - 1 for value in values):
        raise ValueError("launch metadata and padded block coordinates must fit signed int32")
    # End-exclusive formation bounds. Tail lanes are formed but not accessed.
    main_extent = (batch * heads - 1) * head_stride + padded * dim
    history_extent = batch * heads * history_head_stride
    if not all(0 <= extent and 4 * extent <= (1 << 63) - 1
               for extent in (main_extent, history_extent)):
        raise ValueError("formed FP32 displacements must fit signed int64")
    return chunks, head_stride, history_head_stride, main_extent, history_extent


def _check_pointers(tensors, extents):
    for tensor, extent in zip(tensors, extents):
        base = tensor.data_ptr()
        if base < 0 or base % 4 or base + 4 * extent >= 1 << 64:
            raise ValueError("formed FP32 addresses must fit aligned unsigned int64")


def _forward_with_state(q, k, v, store=False, ifcond=False):
    """Internal complete execution; the public autograd wrapper returns O only."""
    if any(not isinstance(t, torch.Tensor) or t.layout != torch.strided for t in (q, k, v)):
        raise ValueError("Q/K/V must be ordinary strided tensors")
    if not isinstance(store, bool) or not isinstance(ifcond, bool):
        raise ValueError("store and ifcond must be bool")
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("Q/K/V must have the same rank-four shape")
    for tensor in (q, k, v):
        if tensor.dtype != torch.float32 or not tensor.is_cuda:
            raise ValueError("Q/K/V must be CUDA FP32 tensors")
        if tensor.device != q.device:
            raise ValueError("Q/K/V must be on the same CUDA device")
        if not tensor.is_contiguous() or tensor.is_conj() or tensor.is_neg():
            raise ValueError("Q/K/V must have resolved contiguous dense storage")
    batch, heads, length, dim = q.shape
    if dim < 16 or dim & (dim - 1):
        raise ValueError("head dimension must be a power of two at least16")
    chunks = triton.cdiv(length, 32)
    h = torch.empty((batch, heads, chunks * dim, dim), dtype=q.dtype, device=q.device)
    o = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if batch == 0 or heads == 0 or length == 0:
        return o, h
    chunks, head_stride, history_head_stride, main_extent, history_extent = _launch_domain(
        batch, heads, length, dim)
    _check_pointers((q, k, v, h, o),
                    (main_extent, main_extent, main_extent, history_extent, main_extent))
    with torch.cuda.device(q.device):
        attention_fwd_kernel[(batch * heads,)](
            q, k, v, h, o, head_stride, dim, 1, history_head_stride, dim,
            length, dim ** -0.5,
            BT=32, BD=dim, NT=chunks, STORE=store, IFCOND=ifcond,
            num_warps=4, num_stages=3 if dim <= 64 else 2,
        )
    return o, h


class AttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, store=False, ifcond=False):
        return _forward_with_state(q, k, v, store, ifcond)[0]
