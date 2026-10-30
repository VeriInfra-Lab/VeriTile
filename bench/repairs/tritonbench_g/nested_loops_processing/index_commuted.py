"""New Python counterpart of the pre-existing Lean integer product candidate.

The authoritative original remains nested_loops_processing_repaired.py unchanged.
Only the second initial input-offset product is commuted; all casts, loops,
loads, unique-lane masks, clamped addresses, barriers and stores are retained.
"""
import triton
import triton.language as tl

@triton.jit
def nested3_index_commuted(in_ptr, out_ptr, stride_m, stride_n, n_elements):
    stride_m = tl.cast(stride_m, tl.int64)
    stride_n = tl.cast(stride_n, tl.int64)
    n_elements = tl.cast(n_elements, tl.int64)
    offs_am = tl.arange(0, 2).to(tl.int64)
    offs_an = tl.arange(0, 2).to(tl.int64)
    a_offsets = offs_am[:, None] * stride_m + stride_n * offs_an[None, :]

    offs_cm = tl.arange(0, 2).to(tl.int64)
    offs_cn = tl.arange(0, 2).to(tl.int64)
    c_offsets = stride_m * offs_cm[:, None] + stride_n * offs_cn[None, :]
    # A contiguous one-column input duplicates lanes (0, 1) and (1, 0).
    # Their cached input words agree; retain only the earlier physical store.
    unique = (offs_cm[:, None] == 0) | (stride_m > 1) | (offs_cn[None, :] == 1)

    for i in range(0, 2):
        a1 = tl.load(in_ptr + tl.minimum(a_offsets, n_elements - 1),
                     mask=a_offsets < n_elements, other=0)
        for j in range(0, 2):
            a_offsets += 2 * stride_n
            a2 = tl.load(in_ptr + tl.minimum(a_offsets, n_elements - 1),
                         mask=a_offsets < n_elements, other=0)
            for k in range(0, 2):
                a_offsets += 2 * stride_n
                a3 = tl.load(in_ptr + tl.minimum(a_offsets, n_elements - 1),
                             mask=a_offsets < n_elements, other=0)
                tl.store(out_ptr + tl.minimum(c_offsets, n_elements - 1), a1,
                         mask=(c_offsets < n_elements) & unique)
                # Uniform CTA barrier orders global writes across source stores.
                tl.debug_barrier()
                c_offsets += 2 * stride_n
                tl.store(out_ptr + tl.minimum(c_offsets, n_elements - 1), a2,
                         mask=(c_offsets < n_elements) & unique)
                # Uniform CTA barrier orders global writes across source stores.
                tl.debug_barrier()
                c_offsets += 2 * stride_n
                tl.store(out_ptr + tl.minimum(c_offsets, n_elements - 1), a3,
                         mask=(c_offsets < n_elements) & unique)
                # Uniform CTA barrier orders global writes across source stores.
                tl.debug_barrier()
                c_offsets += 2 * stride_n
        a_offsets += 2 * stride_n
