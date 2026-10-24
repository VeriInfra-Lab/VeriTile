# Separate repair candidate; the pinned original remains unchanged.
# Original SHA256: dfd772c19ff1b007b3e6c474fa984a460d0e135593cc3d0f05d279c8746997d7
import torch
import triton
import triton.language as tl


@triton.jit
def iv_dependent_matmul_kernel_i64(a_ptr, b_ptr, c_ptr,  #
                                M, N, K,  #
                                stride_am, stride_ak,  #
                                stride_bk, stride_bn,  #
                                stride_cm, stride_cn,  #
                                BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
                                type: tl.constexpr):
    # Widen before products, additions, cdiv, and staged pointer updates.
    M = tl.cast(M, tl.int64)
    N = tl.cast(N, tl.int64)
    K = tl.cast(K, tl.int64)
    stride_am = tl.cast(stride_am, tl.int64)
    stride_ak = tl.cast(stride_ak, tl.int64)
    stride_bk = tl.cast(stride_bk, tl.int64)
    stride_bn = tl.cast(stride_bn, tl.int64)
    stride_cm = tl.cast(stride_cm, tl.int64)
    stride_cn = tl.cast(stride_cn, tl.int64)
    pid = tl.cast(tl.program_id(axis=0), tl.int64)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_SIZE_M + tl.cast(tl.arange(0, BLOCK_SIZE_M), tl.int64)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.cast(tl.arange(0, BLOCK_SIZE_N), tl.int64)) % N
    offs_k = tl.cast(tl.arange(0, BLOCK_SIZE_K), tl.int64)
    a_ptr = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptr = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    a_ptrs = a_ptr
    b_ptrs = b_ptr
    if type == "post_load_two_iters":
        a_ptrs_next = a_ptr + BLOCK_SIZE_K * stride_ak
        b_ptrs_next = b_ptr + BLOCK_SIZE_K * stride_bk
    elif type == "post_load_three_iters":
        a_ptrs_next = a_ptr + BLOCK_SIZE_K * stride_ak
        b_ptrs_next = b_ptr + BLOCK_SIZE_K * stride_bk
        a_ptrs_next_next = a_ptr + 2 * BLOCK_SIZE_K * stride_ak
        b_ptrs_next_next = b_ptr + 2 * BLOCK_SIZE_K * stride_bk

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k64 = tl.cast(k, tl.int64)
        if type == "pre_load":
            a_ptrs = a_ptr + k64 * BLOCK_SIZE_K * stride_ak
            b_ptrs = b_ptr + k64 * BLOCK_SIZE_K * stride_bk
        elif type == "post_pre_mixed":
            a_ptrs = a_ptr + k64 * BLOCK_SIZE_K * stride_ak
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k64 * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k64 * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        if type == "post_load":
            a_ptrs = a_ptr + (k64 + 1) * BLOCK_SIZE_K * stride_ak
            b_ptrs = b_ptr + (k64 + 1) * BLOCK_SIZE_K * stride_bk
        elif type == "post_pre_mixed":
            b_ptrs = b_ptr + (k64 + 1) * BLOCK_SIZE_K * stride_bk
        elif type == "post_load_two_iters":
            a_ptrs = a_ptrs_next
            b_ptrs = b_ptrs_next
            a_ptrs_next = a_ptr + (k64 + 2) * BLOCK_SIZE_K * stride_ak
            b_ptrs_next = b_ptr + (k64 + 2) * BLOCK_SIZE_K * stride_bk
        elif type == "post_load_three_iters":
            a_ptrs = a_ptrs_next
            b_ptrs = b_ptrs_next
            a_ptrs_next = a_ptrs_next_next
            b_ptrs_next = b_ptrs_next_next
            a_ptrs_next_next = a_ptr + (k64 + 3) * BLOCK_SIZE_K * stride_ak
            b_ptrs_next_next = b_ptr + (k64 + 3) * BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.cast(tl.arange(0, BLOCK_SIZE_M), tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.cast(tl.arange(0, BLOCK_SIZE_N), tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def iv_dependent_matmul_wrapper_i64(
    M: int, 
    K: int, 
    N: int, 
    BLOCK_SIZE_M: int, 
    BLOCK_SIZE_N: int, 
    BLOCK_SIZE_K: int, 
    type: str = "pre_load",  # Kernel type for scheduling ("pre_load", "post_load", etc.)
    device: torch.device = "cuda"  # Device to run the test (defaults to "cuda")
):
    dimensions = (M, K, N)
    blocks = (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
    if any(not isinstance(x, int) or isinstance(x, bool) for x in dimensions + blocks):
        raise TypeError("expected integer dimensions and block sizes")
    if min(dimensions) < 0:
        raise ValueError("expected nonnegative dimensions")
    if any(x <= 0 or x > 2**20 or x & (x - 1) for x in blocks):
        raise ValueError("expected positive power-of-two block sizes at most 2**20")
    if type not in ("pre_load", "post_load", "post_pre_mixed", "post_load_two_iters", "post_load_three_iters"):
        raise ValueError("unsupported pointer schedule")
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("expected the original default FP32 allocation profile")
    # Checked scalar ABI, cdiv additions and all three-iteration lookahead indices.
    if max(M + BLOCK_SIZE_M, N + BLOCK_SIZE_N, K + 4 * BLOCK_SIZE_K + 3) >= 2**63:
        raise ValueError("source signed-int64 control envelope exceeded")
    if 4 * max(M * K, K * N, M * N) >= 2**63:
        raise ValueError("source signed-int64 active byte-offset envelope exceeded")
    if triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N) >= 2**31:
        raise ValueError("source signed-int32 program-id envelope exceeded")

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("expected a CUDA device")

    # Generate random input matrices a and b on the specified device
    a = torch.rand((M, K), device=device)
    b = torch.rand((K, N), device=device)

    # Create an empty tensor to store the Triton result
    triton_output = torch.empty((M, N), device=device)

    if M == 0 or N == 0:
        return triton_output

    # Define Triton grid configuration
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    # Set the number of stages based on the kernel type
    num_stages = 4 if type == "post_load_three_iters" else 3

    # Run the Triton kernel
    iv_dependent_matmul_kernel_i64[grid](
        a, b, triton_output, M, N, K,  #
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),  #
        triton_output.stride(0), triton_output.stride(1),  #
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, type=type,  #
        num_stages=num_stages
    )

    # Optionally print the result for inspection
    # print(triton_output)

    return triton_output




##################################################################################################################################################


import torch

# Wrapper for the IV Dependent MatMul test
def test_iv_dependent_matmul_kernel_i64():
    # Define the matrix dimensions and block size.
    M = 256
    K = 256
    N = 256
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    # Create the CUDA device.
    device = torch.device('cuda')

    # Define configurations for all kernel types.
    types = [
        "pre_load",
        "post_load",
        "post_pre_mixed",
        "post_load_two_iters",
        "post_load_three_iters"
    ]

    # Dictionary holding the results for each test case
    results = {}

    # Test each kernel type.
    for i, type in enumerate(types):
        # Call the wrapper to run the Triton kernel.
        triton_output = iv_dependent_matmul_wrapper_i64(
            M=M,
            K=K,
            N=N,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            type=type,
            device=device
        )

        # Check the output size.
        assert triton_output.shape == (M, N), f"Expected output shape {(M, N)} but got {triton_output.shape} for type {type}"

        # Save the results in the dictionary.
        results[f"test_case_{i+1}"] = triton_output

    return results

# Run the test function to check all kernel types.
if __name__ == "__main__":
    result_gold = test_iv_dependent_matmul_kernel_i64()
