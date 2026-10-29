import triton
import triton.language as tl

@triton.jit
def logsumexp_fwd_kernel(
    x: tl.tensor,
    z: tl.tensor,
    scale: tl.float32,
    D: tl.constexpr,
    B: tl.constexpr,
    HAS_SCALE: tl.constexpr
):
    i_n, i_d = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    o_d = i_d * B + tl.arange(0, B)
    m_d = o_d < D

    b_x = tl.load(x + i_n * D + o_d, mask=m_d, other=-float('inf'))
    if HAS_SCALE:
        b_x = b_x * scale
    b_m = tl.max(b_x, 0)
    b_z = tl.log(tl.sum(tl.exp(b_x - b_m), 0)) + b_m
    tl.store(z + i_n * tl.cdiv(D, B) + i_d, b_z)
