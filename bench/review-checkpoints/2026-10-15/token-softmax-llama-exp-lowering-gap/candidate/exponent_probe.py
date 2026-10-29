# Diagnostic instruction graph, explicitly not the original row kernel.
import triton
import triton.language as tl

@triton.jit
def exponent_probe(Input, Scaled, Output, N: tl.constexpr):
    i = tl.arange(0, N)
    x = tl.load(Input + i).to(tl.float32, bitcast=True)
    scaled = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, 0f3FB8AA3B;", constraints="=f,f", args=[x], dtype=tl.float32,
        is_pure=True, pack=1)
    y = tl.inline_asm_elementwise(
        "ex2.approx.f32 $0, $1;", constraints="=f,f", args=[scaled], dtype=tl.float32,
        is_pure=True, pack=1)
    tl.store(Scaled + i, scaled.to(tl.uint32, bitcast=True))
    tl.store(Output + i, y.to(tl.uint32, bitcast=True))
