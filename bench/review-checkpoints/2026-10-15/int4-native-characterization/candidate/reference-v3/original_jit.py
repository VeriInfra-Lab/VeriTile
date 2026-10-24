import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    bs_ptr, bzp_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bsk, stride_bsn,
    stride_bzpk, stride_bzpn,
    group_size,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr
    ):
    """
    assert K % (BLOCK_SIZE_K * SPLIT_K) == 0
    """
    pid = tl.program_id(axis=0)
    pid_sp_k = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # [BLOCK_M, BLOCK_K]
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    # [BLOCK_K, BLOCK_N] but repeated 8 times in N
    b_ptrs = b_ptr + (offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn
    # tl.static_print("shape", a_ptrs, b_ptrs, bs_ptrs, bzp_ptrs)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        # Load the next block of A and B.
        # [BLOCK_K, BLOCK_N] but repeated group_size times in K 
        bs_ptrs = bs_ptr + ((offs_k[:, None] + k * BLOCK_SIZE_K * SPLIT_K) // group_size) * stride_bsk \
            + offs_bn[None, :] * stride_bsn
        # [BLOCK_K, BLOCK_N] but repeated in K and N
        bzp_ptrs = bzp_ptr + ((offs_k[:, None] + k * BLOCK_SIZE_K * SPLIT_K) // group_size) * stride_bzpk \
            + (offs_bn[None, :] // 8) * stride_bzpn
        b_shift_bits = (offs_k[:, None] % 8) * 4 # assert BLOCK_SIZE_K % 8 == 0
        bzp_shift_bits = (offs_bn[None, :] % 8) * 4
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        bs = tl.load(bs_ptrs)
        bzp = tl.load(bzp_ptrs)
        # We accumulate along the K dimension.
        int_b = (b >> b_shift_bits) & 0xF
        int_bzp = (bzp >> bzp_shift_bits) & 0xF
        b = ((int_b - int_bzp) * bs).to(a.dtype)
        accumulator += tl.dot(a, b.to(a.dtype))
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K * SPLIT_K * stride_bk // 8)  # assert BLOCK_SIZE_K % 8 == 0
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = accumulator.to(c_ptr.dtype.element_ty)
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)
