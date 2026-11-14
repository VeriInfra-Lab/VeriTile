"""Actual local source pair and an ordered FP64 oracle/group observer."""
import triton
import triton.language as tl


@triton.jit
def reference_dot(A, B, T):
    g = tl.program_id(0)
    i = tl.arange(0, 32)
    a = tl.load(A + g * 1024 + i[:, None] * 32 + i[None, :])
    b = tl.load(B + g * 1024 + i[:, None] * 32 + i[None, :])
    dot = tl.dot(a, b, input_precision="tf32")
    tl.store(T + g * 1024 + i[:, None] * 32 + i[None, :], dot)


@triton.jit
def reference_add(T, C, P):
    x = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    acc = tl.load(C + x)
    dot = tl.load(T + x)
    tl.store(P + x, acc + dot)


@triton.jit
def candidate(A, B, C, Q):
    g = tl.program_id(0)
    i = tl.arange(0, 32)
    a = tl.load(A + g * 1024 + i[:, None] * 32 + i[None, :])
    b = tl.load(B + g * 1024 + i[:, None] * 32 + i[None, :])
    acc = tl.load(C + g * 1024 + i[:, None] * 32 + i[None, :])
    acc = tl.dot(a, b, acc, input_precision="tf32")
    tl.store(Q + g * 1024 + i[:, None] * 32 + i[None, :], acc)


@triton.jit
def observer(A, B, C, P, Q, O, U, R, SAVE: tl.constexpr):
    g = tl.program_id(0)
    x = tl.arange(0, 1024)
    i, j = x // 32, x % 32
    total = tl.full((1024,), 0.0, tl.float64)
    for k in range(32):
        a = tl.load(A + g * 1024 + i * 32 + k).to(tl.float64)
        b = tl.load(B + g * 1024 + k * 32 + j).to(tl.float64)
        total = total + a * b
    c = tl.load(C + g * 1024 + x).to(tl.float64)
    oracle = total + c
    p = tl.load(P + g * 1024 + x).to(tl.float64)
    q = tl.load(Q + g * 1024 + x).to(tl.float64)
    rounded = tl.minimum(tl.abs(oracle), 3.4028234663852886e38).to(tl.float32)
    bits = rounded.to(tl.uint32, bitcast=True)
    exponent = ((bits >> 23) & 255).to(tl.int32)
    # Construct exact FP64 powers, including FP32's zero/subnormal ULP.
    scale_bits = (tl.maximum(exponent - 150, -149) + 1023).to(tl.uint64) << 52
    ulp = scale_bits.to(tl.float64, bitcast=True)
    finite = (tl.abs(p) < float("inf")) & (tl.abs(q) < float("inf")) & (tl.abs(oracle) < float("inf"))
    valid = tl.sum(finite.to(tl.int32), 0) == 1024
    delta = (q - p) / ulp
    mean = tl.sum(delta, 0) / 1024
    ep = tl.max(tl.abs(p - oracle), 0)
    eq = tl.max(tl.abs(q - oracle), 0)
    tl.store(R + g * 4, tl.where(valid, mean, 0.0))
    tl.store(R + g * 4 + 1, tl.where(valid, ep, 0.0))
    tl.store(R + g * 4 + 2, tl.where(valid, eq, 0.0))
    tl.store(R + g * 4 + 3, valid.to(tl.float64))
    if SAVE:
        tl.store(O + g * 1024 + x, oracle)
        tl.store(U + g * 1024 + x, ulp)
