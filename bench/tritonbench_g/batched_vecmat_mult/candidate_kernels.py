"""Authored from the existing Lean block-first carry-addition candidate.

Only the operands of the reduced-block/carry addition are exchanged.
All original widening casts, broadcasts, reduction and stores remain intact.
"""
import triton
import triton.language as tl

@triton.jit
def batched_vecmat_block_first(
        A,  # shape: [dim_m, dim_k]
        B,  # shape: [dim_m, dim_n, dim_k]
        dim_m, dim_n, dim_k,
        output,
        block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr):
    dim_n = tl.cast(dim_n, tl.int64)
    dim_k = tl.cast(dim_k, tl.int64)
    m_index = tl.program_id(0).to(tl.int64)
    n_index = tl.program_id(1).to(tl.int64)
    output_tile = (m_index * block_m + tl.arange(0, block_m))[:, None] * dim_n \
        + (n_index * block_n + tl.arange(0, block_n))[None, :]

    vecmat = tl.zeros([block_m, block_n], dtype=A.dtype.element_ty)
    k_blocks = dim_k // block_k
    for k_index in range(k_blocks):
        a_tile = (m_index * block_m + tl.arange(0, block_m))[:, None] * dim_k \
            + (k_index * block_k + tl.arange(0, block_k))[None, :]
        a = tl.load(A + a_tile)

        b_tile = (m_index * block_m + tl.arange(0, block_m))[None, :, None] * dim_n * dim_k \
            + (n_index * block_n + tl.arange(0, block_n))[:, None, None] * dim_k \
            + (k_index * block_k + tl.arange(0, block_k))[None, None, :]
        b = tl.load(B + b_tile)

        expanded_a, _ = tl.broadcast(a, b)
        vecmat = tl.trans(tl.sum(expanded_a * b, axis=2)) + vecmat

    tl.store(output + output_tile, vecmat)
