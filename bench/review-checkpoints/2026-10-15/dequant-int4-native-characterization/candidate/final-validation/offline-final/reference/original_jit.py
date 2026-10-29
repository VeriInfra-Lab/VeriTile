import triton
import triton.language as tl

@triton.jit
def matmul4_kernel(
	a_ptr, b_ptr, c_ptr,
	scales_ptr, zeros_ptr,
	M, N, K,
	stride_am, stride_ak,
	stride_bk, stride_bn,
	stride_cm, stride_cn,
	stride_scales_g, stride_scales_n,
	stride_zeros_g, stride_zeros_n,
	groupsize, NO_GROUPS: tl.constexpr,
	BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
	GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute the matrix multiplication C = A x B.
    A is of shape (M, K) float16
    B is of shape (K//8, N) int32
    C is of shape (M, N) float16
    scales is of shape (G, N) float16
    zeros is of shape (G, N//8) int32
    groupsize is an int specifying the size of groups for scales and zeros.
    G is K // groupsize.
    Set NO_GROUPS to groupsize == K, in which case G = 1 and the kernel is more efficient.
    WARNING: This kernel assumes that K is a multiple of BLOCK_SIZE_K.
    WARNING: This kernel assumes that N is a multiple of BLOCK_SIZE_N.
    WARNING: This kernel assumes that groupsize is a multiple of BLOCK_SIZE_K.
    """
    bits = 4
    infearure_per_bits = 8
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)   # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    a_mask = (offs_am[:, None] < M)
    # b_ptrs is set up such that it repeats elements along the K axis 8 times
    b_ptrs = b_ptr + ((offs_k[:, None] // infearure_per_bits) * stride_bk + offs_bn[None, :] * stride_bn)   # (BLOCK_SIZE_K, BLOCK_SIZE_N)
    scales_ptrs = scales_ptr + offs_bn * stride_scales_n   # (BLOCK_SIZE_N,)
    # zeros_ptrs is set up such that it repeats elements along the N axis 8 times
    zeros_ptrs = zeros_ptr + ((offs_bn // infearure_per_bits) * stride_zeros_n)   # (BLOCK_SIZE_N,)
    # shifter is used to extract the 4 bits of each element in the 32-bit word from B and zeros
    shifter = (offs_k % infearure_per_bits) * bits
    zeros_shifter = (offs_bn % infearure_per_bits) * bits
    # If G == 1, scales and zeros are the same for all K, so we can load them once
    if NO_GROUPS:
        # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
        scales = tl.load(scales_ptrs)  # (BLOCK_SIZE_N,)
        zeros = tl.load(zeros_ptrs)  # (BLOCK_SIZE_N,), each element is repeated 8 times, int32	
        # Unpack zeros
        zeros = (zeros >> zeros_shifter) & 0xF  # (BLOCK_SIZE_N,) int32
        # zeros = (zeros + 1) * scales  # (BLOCK_SIZE_N,) float16
        zeros = zeros * scales
    # Now calculate a block of output of shape (BLOCK_SIZE_M, BLOCK_SIZE_N)
    # M is along the batch dimension, N is along the outfeatures dimension, K is along the infeatures dimension
    # So this loop is along the infeatures dimension (K)
    # It's calculating BLOCK_SIZE_M batches in parallel, and for each batch, BLOCK_SIZE_N outfeatures in parallel
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, num_pid_k):
        a = tl.load(a_ptrs, mask=a_mask, other=0.)   # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated
        if not NO_GROUPS:
            g_id = k // (groupsize // BLOCK_SIZE_K)
            ptr = scales_ptrs + g_id * stride_scales_g
            scales = tl.load(ptr)  # (BLOCK_SIZE_N,)
            ptr = zeros_ptrs + g_id * stride_zeros_g   # (BLOCK_SIZE_N,)
            zeros = tl.load(ptr)  # (BLOCK_SIZE_N,), each element is repeated 8 times, int32	
            # Unpack zeros
            zeros = (zeros >> zeros_shifter) & 0xF  # (BLOCK_SIZE_N,) int32
            zeros = (zeros) * scales  # (BLOCK_SIZE_N,) float16	
        # Now we need to unpack b (which is 4-bit values) into 32-bit values
        b = (b >> shifter[:, None]) & 0xF  # Extract the 4-bit values
        b = b * scales[None, :] - zeros[None, :]  # Scale and shift
        # print("data type", a, b)
        accumulator += tl.dot(a, b.to(a.dtype))
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk  
    c = accumulator.to(c_ptr.dtype.element_ty)  
    # Store the result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
