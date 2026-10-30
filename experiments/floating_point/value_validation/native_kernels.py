"""Native operations kept separate from the Lean software reference."""
import triton
import triton.language as tl


@triton.jit
def scalar32_kernel(A, B, OUT, N: tl.constexpr, OP: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + i, i < N, other=0).to(tl.float32, bitcast=True)
    b = tl.load(B + i, i < N, other=0).to(tl.float32, bitcast=True)
    if OP == 'add':
        z = a + b
    elif OP == 'sub':
        z = a - b
    elif OP == 'mul':
        z = a * b
    elif OP == 'div':
        z = a / b
    elif OP == 'div_rn':
        z = tl.div_rn(a, b)
    elif OP == 'neg':
        z = -a
    elif OP == 'abs':
        z = tl.abs(a)
    else:
        tl.static_assert(False, 'unsupported scalar probe')
    tl.store(OUT + i, z.to(tl.int32, bitcast=True), i < N)
