"""Existing head-index product candidate, derived from the repaired source.

Only pid_c * ngroups is commuted; floating code and source masks are retained.
"""
import triton
import triton.language as tl

@triton.jit
def _bmm_chunk_bwd_index_commuted(
    a_ptr, dout_ptr, db_ptr, res_ptr,
    seqlen, chunk_size, K, ngroups,
    stride_a_batch, stride_a_seqlen, stride_a_head, stride_ak,
    stride_dout_batch, stride_dout_chunk, stride_dout_head, stride_dout_csize_m, stride_dout_csize_n,
    stride_db_batch, stride_db_seqlen, stride_db_head, stride_db_k,
    stride_res_batch, stride_res_seqlen, stride_res_head, stride_res_k,
    dot_dtype: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_CS: tl.constexpr,
):
    seqlen = tl.cast(seqlen, tl.int64)
    chunk_size = tl.cast(chunk_size, tl.int64)
    K = tl.cast(K, tl.int64)
    ngroups = tl.cast(ngroups, tl.int64)
    stride_a_batch = tl.cast(stride_a_batch, tl.int64)
    stride_a_seqlen = tl.cast(stride_a_seqlen, tl.int64)
    stride_a_head = tl.cast(stride_a_head, tl.int64)
    stride_ak = tl.cast(stride_ak, tl.int64)
    stride_dout_batch = tl.cast(stride_dout_batch, tl.int64)
    stride_dout_chunk = tl.cast(stride_dout_chunk, tl.int64)
    stride_dout_head = tl.cast(stride_dout_head, tl.int64)
    stride_dout_csize_m = tl.cast(stride_dout_csize_m, tl.int64)
    stride_dout_csize_n = tl.cast(stride_dout_csize_n, tl.int64)
    stride_db_batch = tl.cast(stride_db_batch, tl.int64)
    stride_db_seqlen = tl.cast(stride_db_seqlen, tl.int64)
    stride_db_head = tl.cast(stride_db_head, tl.int64)
    stride_db_k = tl.cast(stride_db_k, tl.int64)
    stride_res_batch = tl.cast(stride_res_batch, tl.int64)
    stride_res_seqlen = tl.cast(stride_res_seqlen, tl.int64)
    stride_res_head = tl.cast(stride_res_head, tl.int64)
    stride_res_k = tl.cast(stride_res_k, tl.int64)
    pid_b = tl.program_id(axis=1).to(tl.int64)
    pid_ch = tl.program_id(axis=2).to(tl.int64)
    pid_c = pid_ch // ngroups
    pid_h = pid_ch - ngroups * pid_c
    num_pid_n = tl.cdiv(K, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0).to(tl.int64) // num_pid_n
    pid_n = tl.program_id(axis=0).to(tl.int64) % num_pid_n

    a_ptr += pid_b * stride_a_batch + pid_c * chunk_size * stride_a_seqlen + pid_h * stride_a_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * stride_dout_chunk + pid_h * stride_dout_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cs = tl.arange(0, BLOCK_SIZE_CS).to(tl.int64)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_csize_n + offs_cs[None, :] * stride_dout_csize_m)
    a_ptrs = a_ptr + (offs_cs[:, None] * stride_a_seqlen + offs_n[None, :] * stride_ak)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for cs in range(0, tl.cdiv(chunk_size_limit, BLOCK_SIZE_CS)):
        dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_cs[None, :] < chunk_size_limit - cs * BLOCK_SIZE_CS), other=0.0).to(dot_dtype)
        a = tl.load(a_ptrs, mask=(offs_cs[:, None] < chunk_size_limit - cs * BLOCK_SIZE_CS) & (offs_n[None, :] < K), other=0.0).to(dot_dtype)
        acc += tl.dot(dout, a)
        dout_ptrs += BLOCK_SIZE_CS * stride_dout_csize_m
        a_ptrs += BLOCK_SIZE_CS * stride_a_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if HAS_RESIDUAL:
        res_ptr += pid_b * stride_res_batch + pid_c * chunk_size * stride_res_seqlen + pid_h * stride_res_head
        res_ptrs = res_ptr + (offs_m[:, None] * stride_res_seqlen + offs_n[None, :] * stride_res_k)
        res = tl.load(res_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < K)).to(tl.float32)
        acc += res
    db = acc.to(db_ptr.dtype.element_ty)

    db_ptr += pid_b * stride_db_batch + pid_c * chunk_size * stride_db_seqlen + pid_h * stride_db_head
    db_ptrs = db_ptr + (offs_m[:, None] * stride_db_seqlen + offs_n[None, :] * stride_db_k)
    tl.store(db_ptrs, db, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < K))
