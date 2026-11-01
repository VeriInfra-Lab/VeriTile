"""Direct-output candidate: inline the existing FP16 cast into the store.

The FP32 route has no explicit source cast. This candidate preserves the
original FP16 conversion, dot instructions, strides and block pointers.
"""
import triton
import triton.language as tl

@triton.jit
def matmul_tma_direct_output(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, OUTPUT_F16: tl.constexpr):
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

    # Store the result
    if OUTPUT_F16:
        tl.store(c_block_ptr, c.to(tl.float16))
    else:
        tl.store(c_block_ptr, c)
