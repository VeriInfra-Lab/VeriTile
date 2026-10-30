"""Separate repair candidate; original SHA256:
8f67a536f4c7850f6025debc90d692e3d8e007159e92df38ea144c20eac77b1f.

All nine cumulative autotune configurations and all floating statements are
unchanged. Kernel indices and tile bases use signed 64-bit arithmetic, and
block-pointer offsets are zero to avoid their implicit signed-32-bit narrowing. The adapter pads only operands whose logical shapes need extra coordinates.
Aligned contiguous operands are read directly. Full tile padding makes every launched block access defined;
partial chunk gates repeat the last valid cumulative gate. Returned tensors
have the original logical shapes. Optional final state is copied through the
original output view only after all kernels have finished reading input views.

The host splits a flattened launch grid into legal one-dimensional batches.
This is a separate repair candidate; its proof scope is recorded in README.md.
"""

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BS': 16}, num_warps=2),
        triton.Config({'BS': 16}, num_warps=4),
        triton.Config({'BS': 16}, num_warps=8),
        triton.Config({'BS': 32}, num_warps=2),
        triton.Config({'BS': 32}, num_warps=4),
        triton.Config({'BS': 32}, num_warps=8),
        triton.Config({'BS': 64}, num_warps=2),
        triton.Config({'BS': 64}, num_warps=4),
        triton.Config({'BS': 64}, num_warps=8),
    ],
    key=['S']
)
@triton.jit
def chunk_gated_abc_fwd_kernel_cum(
    s,
    o,
    s_s_h,
    s_s_t,
    s_s_d,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BH: tl.constexpr,
    PROGRAM_BASE: tl.constexpr,
):
    linear = tl.cast(tl.program_id(0), tl.int64) + PROGRAM_BASE
    NS: tl.constexpr = S // BS
    NT: tl.constexpr = T // BT
    # Some autotune configurations use fewer feature blocks than BS=16.
    # A dummy launch for an exhausted configuration has no memory effects.
    if linear < BH * NT * NS:
        i_s = linear % NS
        i_t = (linear // NS) % NT
        i_bh = linear // (NS * NT)
        o_i = tl.arange(0, BT)
        m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.).to(tl.float32)

        tile_base = i_bh * s_s_h + i_t * BT * s_s_t + i_s * BS * s_s_d
        p_s = tl.make_block_ptr(s + tile_base, (BT, BS), (s_s_t, s_s_d), (0, 0), (BT, BS), (1, 0))
        p_o = tl.make_block_ptr(o + tile_base, (BT, BS), (s_s_t, s_s_d), (0, 0), (BT, BS), (1, 0))
        # [BT, BS]
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        b_o = tl.dot(m_s, b_s, allow_tf32=False)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_gated_abc_fwd_kernel_h(
    k,
    v,
    g,
    h,
    h0,
    ht,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    s_h_h,
    s_h_t,
    s_h_d,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    GATEK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    PROGRAM_BASE: tl.constexpr,
):
    linear = tl.cast(tl.program_id(0), tl.int64) + PROGRAM_BASE
    NV: tl.constexpr = V // BV
    NK: tl.constexpr = K // BK
    i_v = linear % NV
    i_k = (linear // NV) % NK
    i_bh = linear // (NV * NK)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(h0 + i_bh * K * V + i_k * BK * V + i_v * BV, (BK, BV), (V, 1), (0, 0), (BK, BV), (1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)
    for i_t in range(tl.cast(0, tl.int64), tl.cast(NT, tl.int64), tl.cast(1, tl.int64)):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h + i_k * BK * s_k_d + i_t * BT * s_k_t, (BK, BT), (s_k_d, s_k_t), (0, 0), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h + i_t * BT * s_v_t + i_v * BV * s_v_d, (BT, BV), (s_v_t, s_v_d), (0, 0), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V + i_k * BK * s_h_t + i_v * BV * s_h_d, (BK, BV), (s_h_t, s_h_d), (0, 0), (BK, BV), (1, 0))

        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        if GATEK:
            p_g = tl.make_block_ptr(g + i_bh * s_k_h + i_k * BK * s_k_d + i_t * BT * s_k_t, (BK, BT), (s_k_d, s_k_t), (0, 0), (BK, BT), (0, 1))
            p_gn = tl.make_block_ptr(g + i_bh * s_k_h + (i_t * BT + BT - 1) * s_k_t + i_k * BK * s_k_d, (BK,), (s_k_d,), (0,), (BK,), (0,))
            # [BK,]
            b_gn = tl.load(p_gn, boundary_check=(0,))
            # [BK, BV]
            b_h *= tl.exp(b_gn)[:, None]
            # [BK, BT]
            b_g = tl.load(p_g, boundary_check=(0, 1))
            b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(b_k.dtype)
        else:
            p_g = tl.make_block_ptr(g + i_bh * s_v_h + i_t * BT * s_v_t + i_v * BV * s_v_d, (BT, BV), (s_v_t, s_v_d), (0, 0), (BT, BV), (1, 0))
            p_gn = tl.make_block_ptr(g + i_bh * s_v_h + (i_t * BT + BT - 1) * s_v_t + i_v * BV * s_v_d, (BV,), (s_v_d,), (0,), (BV,), (0,))
            # [BV,]
            b_gn = tl.load(p_gn, boundary_check=(0,))
            # [BK, BV]
            b_h *= tl.exp(b_gn)[None, :]
            # [BT, BV]
            b_g = tl.load(p_g, boundary_check=(0, 1))
            b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(b_v.dtype)
        # [BK, BV]
        b_h += tl.dot(b_k, b_v, allow_tf32=False)

    if STORE_FINAL_STATE:
        p_h = tl.make_block_ptr(ht + i_bh * K * V + i_k * BK * V + i_v * BV, (BK, BV), (V, 1), (0, 0), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))



_MAX_GRID_X = 2147483647


def _launch_batches(total):
    for base in range(0, total, _MAX_GRID_X):
        yield base, min(total - base, _MAX_GRID_X)


def _dimensions(*values):
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("dimensions must be nonnegative Python integers")


def _block(*values):
    if any(type(value) is not int or value < 16 or value & (value - 1) for value in values):
        raise ValueError("dot block sizes must be powers of two, at least 16")


def _tensor(tensor, shape, name, device=None):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have logical shape {tuple(shape)}")
    if (tensor.layout != torch.strided or tensor.dtype != torch.float32 or not tensor.is_cuda
            or not tensor.is_contiguous() or tensor.is_neg() or tensor.is_conj()):
        raise ValueError(f"{name} must be an ordinary contiguous FP32 CUDA tensor")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must be on the same device as the other inputs")


def _addresses(*tensors):
    for tensor in tensors:
        if tensor is None:
            continue
        pointer = tensor.data_ptr()
        if pointer < 0 or pointer % 4 or pointer + 4*tensor.numel() >= 2**64:
            raise ValueError("invalid FP32 pointer alignment or address span")


def _sizes(*counts):
    if any(4*count >= 2**63 for count in counts):
        raise ValueError("storage byte extent exceeds signed 64 bits")


def _materialize(tensor, shape):
    # Reuse aligned contiguous inputs. A padding allocation is needed only
    # when the complete-tile source requires extra time or feature coordinates.
    if tuple(tensor.shape) == tuple(shape):
        return tensor
    output = torch.zeros(shape, dtype=torch.float32, device=tensor.device)
    _addresses(output)
    output[tuple(slice(0, size) for size in tensor.shape)].copy_(tensor)
    return output


def fwd_pre_repaired(g, B, H, T, S, BT):
    """Chunk-local cumulative gates; g has logical shape [B,H,T,S]."""
    _dimensions(B, H, T, S)
    _block(BT)
    if max(BT*BT, BT*64) > 2**20:
        raise ValueError("cumulative tensor tile exceeds Triton's element limit")
    _tensor(g, (B, H, T, S), "g")
    if B * H * T * S == 0:
        return torch.empty((B, H, T, S), dtype=torch.float32, device=g.device)
    NT = triton.cdiv(T, BT)
    PT, PS = NT * BT, triton.cdiv(S, 64) * 64
    _sizes(B*H*PT*PS)
    _addresses(g)
    with torch.cuda.device(g.device):
        g_org = _materialize(g, (B, H, PT, PS))
        g_out = torch.empty((B, H, PT, PS), dtype=torch.float32, device=g.device)
        _addresses(g_org, g_out)
        # BS16 determines the outer batches. Exhausted BS32/BS64 configurations
        # execute a guarded dummy program with no memory accesses.
        for program_base, _ in _launch_batches(B * H * NT * (PS // 16)):
            def grid(meta):
                total = B * H * NT * (PS // meta['BS'])
                return (max(1, min(_MAX_GRID_X, total - program_base)),)
            chunk_gated_abc_fwd_kernel_cum[grid](
                g_org, g_out, PT*PS, PS, 1,
                T=PT, S=PS, BT=BT, BH=B*H, PROGRAM_BASE=program_base
            )
        return g_out[:, :, :T, :S].contiguous()


def fwd_inner_repaired(q, k, v, g, B, H, T, K, V, BT, BK, BV,
                       gatek=False, h0=None, ht=None):
    """Return history [B,H,ceil(T/BT)*K,V], and copy final state into ht.

    K/V/G use logical [B,H,time,feature] axes. Read-only inputs may alias;
    caller ht may alias any input because copy-back occurs after all launches.
    """
    _dimensions(B, H, T, K, V)
    _block(BT, BK, BV)
    if max(BT*BK, BT*BV, BK*BV) > 2**20:
        raise ValueError("recurrent tensor tile exceeds Triton's element limit")
    if type(gatek) is not bool:
        raise TypeError("gatek must be a bool")
    _tensor(q, (B, H, T, K), "q")
    _tensor(k, (B, H, T, K), "k", q.device)
    _tensor(v, (B, H, T, V), "v", q.device)
    _tensor(g, (B, H, T, K if gatek else V), "g", q.device)
    if h0 is not None:
        _tensor(h0, (B, H, K, V), "h0", q.device)
    if ht is not None:
        _tensor(ht, (B, H, K, V), "ht", q.device)
    NT = triton.cdiv(T, BT)
    _sizes(B*H*T*K, B*H*T*V, B*H*K*V, B*H*NT*K*V)
    _addresses(q, k, v, g, h0, ht)
    with torch.cuda.device(q.device):
        if B * H * T * K * V == 0:
            if ht is not None:
                # Preserve FP32 zero-plus-seed, including signed-zero behavior.
                final = torch.zeros((B, H, K, V), dtype=torch.float32, device=q.device)
                if h0 is not None:
                    final = final + h0
                ht.copy_(final)
            return q.new_empty(B, H, NT*K, V)
        PT, PK, PV = NT * BT, triton.cdiv(K, BK) * BK, triton.cdiv(V, BV) * BV
        _sizes(B*H*PT*PK, B*H*PT*PV, B*H*NT*PK*PV, B*H*PK*PV)
        key = _materialize(k, (B, H, PT, PK))
        value = _materialize(v, (B, H, PT, PV))
        gate = _materialize(g, (B, H, PT, PK if gatek else PV))
        if T < PT:
            # Only padded rows are written; the source endpoint row is disjoint.
            gate[:, :, T:, :].copy_(gate[:, :, T-1:T, :])
        seed = _materialize(h0, (B, H, PK, PV)) if h0 is not None else None
        final = torch.empty((B, H, PK, PV), dtype=torch.float32, device=q.device) if ht is not None else None
        history = q.new_empty(B, H, NT*PK, PV)
        _addresses(key, value, gate, seed, final, history)
        for program_base, grid_size in _launch_batches(B*H*(PK//BK)*(PV//BV)):
            chunk_gated_abc_fwd_kernel_h[(grid_size,)](
                key, value, gate, history, seed, final,
                PT*PK, PK, 1, PT*PV, PV, 1, NT*PK*PV, PV, 1,
                T=PT, K=PK, V=PV, BT=BT, BK=BK, BV=BV, NT=NT,
                GATEK=gatek, USE_INITIAL_STATE=seed is not None,
                STORE_FINAL_STATE=final is not None, PROGRAM_BASE=program_base,
                num_warps=4 if BK == 64 else 2, num_stages=1
            )
        if ht is not None:
            ht.copy_(final[:, :, :K, :V])
        return history.view(B,H,NT,PK,PV)[:, :, :, :K, :V].reshape(B,H,NT*K,V).contiguous()
