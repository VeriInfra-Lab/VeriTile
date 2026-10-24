import triton
import triton.language as tl
@triton.jit
def kernel(
    M: tl.tensor,
    Out: tl.tensor,
    matrix_stridex: tl.int32,
    matrix_stridey: tl.int32,
    out_stridex: tl.int32,
    out_stridey: tl.int32,
    SIZE_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    size_m_arange = tl.arange(0, SIZE_M)
    d_head_arange = tl.arange(0, D_HEAD)
    matrix_ptr = M + size_m_arange[:, None] * matrix_stridex + d_head_arange[None, :] * matrix_stridey
    out_ptr = Out + d_head_arange[:, None] * out_stridex + size_m_arange[None, :] * out_stridey
    matrix = tl.load(matrix_ptr)
    tl.store(out_ptr, tl.trans(matrix))
