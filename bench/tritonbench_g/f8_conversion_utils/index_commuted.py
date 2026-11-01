"""Python counterparts of the existing Lean integer-index candidates.

Authored during source migration; not historical benchmark reference source.
The only transformation commutes pid * BLOCK_SIZE in offs. Decode keeps both
stores, and the caller supplies the same six existing dtype profiles.
"""
import triton
import triton.language as tl

@triton.jit
def kernel_f8_to_f16_index_commuted(Y, X, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = BLOCK_SIZE * pid + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    tl.store(Y + offs, x, mask=mask)
    tl.store(Y + offs, x, mask=mask)

@triton.jit
def kernel_f16_to_f8_index_commuted(Y, X, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = BLOCK_SIZE * pid + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    tl.store(Y + offs, x, mask=mask)

