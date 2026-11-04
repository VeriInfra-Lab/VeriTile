"""Existing address commutations applied to the selected repaired JIT bodies.
Generated from softmax_flaggems.py; no floating expression or store is changed.
"""
import triton
import triton.language as tl
from softmax_flaggems import prev_multiple_of

@triton.jit
def softmax_kernel_non_inner_address_commuted(
    output_ptr,
    input_ptr,
    M: tl.int64,
    N: tl.int64,
    K: tl.int64,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid_k = tl.program_id(1).to(tl.int64)
    pid_m = tl.program_id(0).to(tl.int64)

    k_offsets = TILE_K * pid_k + tl.arange(0, TILE_K).to(tl.int64)

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N).to(tl.int64)
        offset = pid_m * N * K + n_offsets[:, None] * K + k_offsets
        mask = (n_offsets[:, None] < N) & (k_offsets < K)
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        m = tl.max(inp, 0)
        e = tl.exp(inp - m[None, :])
        z = tl.sum(e, 0)
        out = e / z
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
    else:
        m = tl.full([TILE_N, TILE_K], value=-340282346638528859811704183484516925440.0, dtype=tl.float32)
        z = tl.full([TILE_N, TILE_K], value=0.0, dtype=tl.float32)

        # specialization does not improve performance inn this example, as tested
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N).to(tl.int64)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets
            mask = (n_offsets[:, None] < N) & (k_offsets < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
            m_new = tl.maximum(m, inp)
            alpha = tl.exp(m - m_new)
            z = z * alpha + tl.exp(inp - m_new)
            m = m_new

        m_reduced = tl.max(m, 0)  # (TILE_K,)
        z = tl.sum(z * tl.exp(m - m_reduced[None, :]), 0)  # (TILE_K, )
        m = m_reduced

        # specialization does not improve performance inn this example, as tested
        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N).to(tl.int64)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets
            mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
            o = tl.exp(inp - m[None, :]) / z[None, :]
            tl.store(output_ptr + offsets, o, mask=mask)

@triton.jit
def softmax_kernel_inner_address_commuted(
    output_ptr,
    input_ptr,
    M: tl.int64,
    N: tl.int64,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N).to(tl.int64)
        offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(
            output_ptr.dtype.element_ty
        )
        m = tl.max(inp, 0)
        e = tl.exp(inp - m)
        z = tl.sum(e, 0)
        out = e / z
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
    else:
        m = tl.full([TILE_N], value=-340282346638528859811704183484516925440.0, dtype=tl.float32)
        z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
        input_ptr += N * pid_m
        output_ptr += N * pid_m

        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N).to(tl.int64)
            inp = tl.load(input_ptr + n_offsets)
            m_new = tl.maximum(m, inp)
            z = z * tl.exp(m - m_new) + tl.exp(inp - m_new)
            m = m_new
        # specialize the last iteration
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N).to(tl.int64)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf"))
            m_new = tl.maximum(m, inp)
            z = z * tl.exp(m - m_new) + tl.exp(inp - m_new)
            m = m_new

        m_reduced = tl.max(m, 0)
        z = tl.sum(z * tl.exp(m - m_reduced), 0)
        m = m_reduced

        previous_multiple = prev_multiple_of(N, TILE_N)
        # specialize the first iteration
        for start_n in range(0, TILE_N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N).to(tl.int64)
            mask = n_offsets < N
            inp = tl.load(
                input_ptr + n_offsets,
                mask=mask,
                other=-float("inf"),
                eviction_policy="evict_first",
            )
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o, mask=mask)
        for start_n in range(TILE_N, N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N).to(tl.int64)
            inp = tl.load(input_ptr + n_offsets, eviction_policy="evict_first")
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o)

@triton.jit
def softmax_backward_kernel_non_inner_address_commuted(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M: tl.int64,
    N: tl.int64,
    K: tl.int64,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1).to(tl.int64)
    offsets_k = TILE_K * pid_k + tl.arange(0, TILE_K).to(tl.int64)

    if ONE_TILE_PER_CTA:
        offsets_n = tl.arange(0, TILE_N).to(tl.int64)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        mask = (offsets_n < N)[:, None] & (offsets_k < K)
        out_tile = tl.load(out_ptr + offsets, mask=mask, other=0)
        out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask, other=0)
        scale = tl.sum(out_tile * out_grad_tile, axis=0)
        in_grad_tile = out_tile * (out_grad_tile - scale[None, :])
        tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
    else:
        offsets_n = tl.arange(0, TILE_N).to(tl.int64)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        scale = tl.zeros([TILE_N, TILE_K], dtype=tl.float32)
        for _ in range(0, N, TILE_N):
            mask = (offsets_n < N)[:, None] & (offsets_k < K)
            out_tile = tl.load(out_ptr + offsets, mask=mask, other=0)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask, other=0)
            scale += out_tile * out_grad_tile
            offsets_n += TILE_N
            offsets += TILE_N * K
        scale = tl.sum(scale, axis=0)  # (TILE_K)

        offsets_n = tl.arange(0, TILE_N).to(tl.int64)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        for _ in range(0, N, TILE_N):
            mask = (offsets_n < N)[:, None] & (offsets_k < K)
            out_tile = tl.load(out_ptr + offsets, mask=mask, other=0)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask, other=0)
            in_grad_tile = out_tile * (out_grad_tile - scale[None, :])
            tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
            offsets_n += TILE_N
            offsets += TILE_N * K

@triton.jit
def softmax_backward_kernel_inner_address_commuted(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M: tl.int64,
    N: tl.int64,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    m_offsets = TILE_M * pid_m + tl.arange(0, TILE_M).to(tl.int64)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N).to(tl.int64)
        offsets = m_offsets[:, None] * N + n_offsets
        mask = (m_offsets[:, None] < M) & (n_offsets < N)
        out_tile = tl.load(out_ptr + offsets, mask=mask, other=0)
        out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask, other=0)
        scale = tl.sum(out_tile * out_grad_tile, 1)
        in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
        tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
    else:
        scale = tl.zeros([TILE_M, TILE_N], dtype=tl.float32)

        n_offsets = tl.arange(0, TILE_N).to(tl.int64)
        offsets = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            out_tile = tl.load(
                out_ptr + offsets, mask=mask, eviction_policy="evict_last", other=0
            )
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask, other=0)
            scale += out_tile * out_grad_tile
            n_offsets += TILE_N
            offsets += TILE_N
        scale = tl.sum(scale, 1)  # (TILE_M,)

        n_offsets = tl.arange(0, TILE_N).to(tl.int64)
        offsets = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            out_tile = tl.load(
                out_ptr + offsets, mask=mask, eviction_policy="evict_first", other=0
            )
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask, other=0)
            in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
            tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
            n_offsets += TILE_N
            offsets += TILE_N
