"""Native counterpart of the existing quotient/remainder address candidate.

The production wrapper remains in cache_transform_triton_repaired.py. Pointer
views have uint8, int64 and uint8 elements, respectively; all sizes are constexpr.
"""
import triton
import triton.language as tl


@triton.jit
def gather_cache_bytes_quotient_remainder(Cache, Rows, Output, COUNT: tl.constexpr,
                                          ROW_BYTES: tl.constexpr, START: tl.constexpr,
                                          BLOCK: tl.constexpr):
    idx = tl.program_id(0).to(tl.uint64) * BLOCK + tl.arange(0, BLOCK).to(tl.uint64) + START
    active = idx < COUNT
    safe = tl.where(active, idx, 0)
    row = tl.load(Rows + safe // ROW_BYTES).to(tl.uint64)
    offset = row * ROW_BYTES + (safe - (safe // ROW_BYTES) * ROW_BYTES)
    value = tl.load(Cache + offset, mask=active, other=0)
    tl.store(Output + safe, value, mask=active)
