
import torch
import triton
import triton.language as tl

inv_ln2 = tl.constexpr(1.44269504)

# Forward decay cumulative sum kernel
@triton.jit
def fwd_decay_cumsum(
    g,
    g_o,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    B,
    H,
    T,
    scale,
    BT: tl.constexpr,
    BK: tl.constexpr,
    DK: tl.constexpr
):
    i_k = tl.program_id(0).to(tl.int64)
    i_c = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    p_g = g + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_go = g_o + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    cum_decay = tl.zeros([BK], dtype=tl.float32)
    mask = (i_k * BK + tl.arange(0, BK)) < DK

    for i in range(tl.cast(BT, tl.int64)):
        _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        cum_decay += _g * inv_ln2
        tl.store(p_go, cum_decay.to(p_go.dtype.element_ty), mask=mask)
        p_g += DK
        p_go += DK

# Prepare qg and kg kernel
@triton.jit
def prepare_qg_kg(
    q,
    k,
    g,
    qg,
    kg,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    B,
    H,
    T,
    scale,
    BT: tl.constexpr,
    BK: tl.constexpr,
    DK: tl.constexpr
):

    i_k = tl.program_id(0).to(tl.int64)
    i_c = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    p_q = q + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_g = g + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_qg = qg + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_kg = kg + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)

    mask = (i_k * BK + tl.arange(0, BK)) < DK

    last_decay = tl.load(g + i_bh * s_qk_h + (i_c * BT + BT - 1) * DK + i_k * BK + tl.arange(0, BK))

    for i in range(tl.cast(BT, tl.int64)):
        _q = tl.load(p_q, mask=mask, other=0)
        _k = tl.load(p_k, mask=mask, other=0)
        _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        _q *= tl.math.exp2(_g) * scale
        _k *= tl.math.exp2(last_decay - _g)
        tl.store(p_kg, _k.to(p_kg.dtype.element_ty), mask=mask)
        tl.store(p_qg, _q.to(p_qg.dtype.element_ty), mask=mask)
        p_q += DK
        p_g += DK
        p_k += DK
        p_kg += DK
        p_qg += DK

# Backward decay global cumulative sum kernel
@triton.jit
def bwd_decay_global_cumsum(
    dq_inner,
    dq_inter,
    dk_inner,
    dk_inter,
    q, k, g, dg,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    B,
    H,
    T,
    scale,
    BT: tl.constexpr,
    BK: tl.constexpr,
    DK: tl.constexpr
):
    i_k = tl.program_id(0).to(tl.int64)
    i_c = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_g = g + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dg = dg + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dq_inner = dq_inner + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dk_inner = dk_inner + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dq_inter = dq_inter + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dk_inter = dk_inter + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    cum_grad_dg = tl.zeros([BK], dtype=tl.float32)
    mask = (i_k * BK + tl.arange(0, BK)) < DK
    last_g = tl.zeros([BK], dtype=tl.float32)
    for j in range(tl.cast(BT - 1, tl.int64), tl.cast(-1, tl.int64), -1):
        _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        if j == (BT-1):
            last_g = _g
        _dq1 = tl.load(p_dq_inner, mask=mask, other=0)
        _dq2 = tl.load(p_dq_inter, mask=mask, other=0)
        _dq2 *= tl.math.exp2(_g)
        _dq = _dq1 + _dq2
        tl.store(p_dq_inter, _dq, mask=mask)
        _dk1 = tl.load(p_dk_inner, mask=mask, other=0)
        _dk2 = tl.load(p_dk_inter, mask=mask, other=0)
        _dk2 *= tl.math.exp2(last_g - _g)
        _dk = _dk1 + _dk2
        tl.store(p_dk_inter, _dk, mask=mask)
        _q = tl.load(p_q, mask=mask, other=0)
        _k = tl.load(p_k, mask=mask, other=0)
        _dg = _dq * _q - _dk * _k
        cum_grad_dg += _dg
        tl.store(p_dg, cum_grad_dg.to(p_dg.dtype.element_ty), mask=mask)
        p_g -= DK
        p_k -= DK
        p_q -= DK
        p_dq_inner -= DK
        p_dk_inner -= DK
        p_dq_inter -= DK
        p_dk_inter -= DK
        p_dg -= DK

# A launch uses independent packed snapshots of every logical read role.
# Full chunks and the short final chunk retain the original numerical loop.
def _dimensions(B, H, T, BT, BK, DK):
    if min(B, H, T, DK) < 0:
        raise ValueError("tensor dimensions must be nonnegative")
    if BT <= 0 or BK <= 0:
        raise ValueError("tile dimensions must be positive")
    return triton.cdiv(DK, BK) * BK


def _snapshot(x, B, H, T, DK, width):
    logical = x
    packed = torch.zeros((B, H, T, width), dtype=x.dtype, device=x.device)
    packed[..., :DK].copy_(logical)
    return packed


def _output(x, B, H, T, width):
    return torch.empty((B, H, T, width), dtype=x.dtype, device=x.device)


def _launch(kernel, buffers, B, H, T, scale, BT, BK, width):
    with torch.cuda.device(buffers[0].device):
        complete, remainder = divmod(T, BT)
        head_stride = T * width
        if complete:
            kernel[(width // BK, complete, B * H)](
                *buffers, head_stride, width, 1, B, H, T, scale,
                BT=BT, BK=BK, DK=width)
        if remainder:
            tail = tuple(x[:, :, complete * BT:, :] for x in buffers)
            kernel[(width // BK, 1, B * H)](
                *tail, head_stride, width, 1, B, H, remainder, scale,
                BT=remainder, BK=BK, DK=width)


def _writeback(destination, packed, DK):
    destination.copy_(packed[..., :DK])


def _check_inputs(tensors, B, H, T, scale, BT, BK, DK):
    if any(not isinstance(x, torch.Tensor) for x in tensors):
        raise TypeError("all tensor arguments must be Tensors")
    if any(not isinstance(d, int) for d in (B, H, T, BT, BK, DK)):
        raise TypeError("dimensions and tile sizes must be integers")
    if min(B, H, T, DK) < 0 or BT <= 0 or BK <= 0:
        raise ValueError("dimensions must be nonnegative and tile sizes positive")
    if BK & (BK - 1) or BK > 1048576:
        raise ValueError("BK must be a supported power-of-two arange extent")
    if not isinstance(scale, float):
        raise TypeError("scale must be a Python float")
    if any(x.layout != torch.strided for x in tensors):
        raise ValueError("all tensors must use strided layout")
    if any(x.dtype != torch.float32 for x in tensors):
        raise TypeError("all tensors must be FP32")
    if any(x.shape != (B, H, T, DK) for x in tensors):
        raise ValueError("tensor shapes must agree with B, H, T, DK")
    if any(not x.is_cuda for x in tensors):
        raise ValueError("all tensors must be CUDA")
    if any(x.device != tensors[0].device for x in tensors):
        raise ValueError("all tensors must use the same CUDA device")
    if any(not x.is_contiguous() for x in tensors):
        raise ValueError("all tensors must be contiguous")
    if any(x.is_neg() for x in tensors):
        raise ValueError("all tensors must have no unresolved negative view bit")
    if any(x.numel() > 0 and x.data_ptr() % 4 != 0 for x in tensors):
        raise ValueError("nonempty tensors must have four-byte-aligned data pointers")
    if B * H * T * DK > 0 and ((DK + BK - 1) // BK >= 2**31 or T // BT > 65535 or B * H > 65535):
        raise ValueError("the source grid dimensions exceed the CUDA launch limits")


# The public wrappers retain their supplied output objects and None return.
def launch_fwd_decay_cumsum(g, g_o, B, H, T, scale, BT, BK, DK):
    _check_inputs((g, g_o), B, H, T, scale, BT, BK, DK)
    width = _dimensions(B, H, T, BT, BK, DK)
    if B * H * T * DK == 0:
        return
    source = _snapshot(g, B, H, T, DK, width)
    output = _output(g_o, B, H, T, width)
    _launch(fwd_decay_cumsum, (source, output), B, H, T, scale, BT, BK, width)
    _writeback(g_o, output, DK)


def launch_prepare_qg_kg(q, k, g, qg, kg, B, H, T, scale, BT, BK, DK):
    _check_inputs((q, k, g, qg, kg), B, H, T, scale, BT, BK, DK)
    width = _dimensions(B, H, T, BT, BK, DK)
    if B * H * T * DK == 0:
        return
    qs = _snapshot(q, B, H, T, DK, width)
    ks = _snapshot(k, B, H, T, DK, width)
    gs = _snapshot(g, B, H, T, DK, width)
    qo = _output(qg, B, H, T, width)
    ko = _output(kg, B, H, T, width)
    _launch(prepare_qg_kg, (qs, ks, gs, qo, ko), B, H, T, scale, BT, BK, width)
    _writeback(kg, ko, DK)
    _writeback(qg, qo, DK)


def launch_bwd_decay_global_cumsum(dq_inner, dq_inter, dk_inner, dk_inter, q, k, g, dg, B, H, T, scale, BT, BK, DK):
    _check_inputs((dq_inner, dq_inter, dk_inner, dk_inter, q, k, g, dg), B, H, T, scale, BT, BK, DK)
    width = _dimensions(B, H, T, BT, BK, DK)
    if B * H * T * DK == 0:
        return
    dqi = _snapshot(dq_inner, B, H, T, DK, width)
    dqo = _snapshot(dq_inter, B, H, T, DK, width)
    dki = _snapshot(dk_inner, B, H, T, DK, width)
    dko = _snapshot(dk_inter, B, H, T, DK, width)
    qs = _snapshot(q, B, H, T, DK, width)
    ks = _snapshot(k, B, H, T, DK, width)
    gs = _snapshot(g, B, H, T, DK, width)
    go = _output(dg, B, H, T, width)
    _launch(bwd_decay_global_cumsum, (dqi, dqo, dki, dko, qs, ks, gs, go), B, H, T, scale, BT, BK, width)
    _writeback(dq_inter, dqo, DK)
    _writeback(dk_inter, dko, DK)
    _writeback(dg, go, DK)



