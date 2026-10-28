"""Historical redundant FP32 output cast, now explicitly a candidate.

The original Python has no FP32 cast to remove. This preserves that existing
local conversion comparison in the opposite orientation, without attributing
an invented instruction to the original JIT.
"""
import triton
import triton.language as tl

@triton.jit
def matmul_tma_explicit_output(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        OUTPUT_F16: tl.constexpr
):
    # Create block pointers for A, B, and C matrices
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak), offsets=(0, 0),
                                    block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn), offsets=(0, 0),
                                    block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))
    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn), offsets=(0, 0),
                                    block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    # Load A and B blocks
    a = tl.load(a_block_ptr)
    b = tl.load(b_block_ptr)

    # Compute matrix product
    c = tl.dot(a, b)
    # Optionally convert the result to float16
    if OUTPUT_F16:
        c = c.to(tl.float16)
    else:
        c = c.to(c_ptr.type.element_ty)

    # Store the result
    tl.store(c_block_ptr, c)
