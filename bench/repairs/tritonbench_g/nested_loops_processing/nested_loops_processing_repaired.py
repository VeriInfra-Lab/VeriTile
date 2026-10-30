import torch
import triton
import triton.language as tl


@triton.jit
def nested3_repaired(in_ptr, out_ptr, stride_m, stride_n, n_elements):
    stride_m = tl.cast(stride_m, tl.int64)
    stride_n = tl.cast(stride_n, tl.int64)
    n_elements = tl.cast(n_elements, tl.int64)
    offs_am = tl.arange(0, 2).to(tl.int64)
    offs_an = tl.arange(0, 2).to(tl.int64)
    a_offsets = offs_am[:, None] * stride_m + offs_an[None, :] * stride_n

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


def wrapper_nested3_repaired(n_rows, n_cols):
    if type(n_rows) is not int or type(n_cols) is not int:
        raise TypeError("dimensions must be Python integers")
    if not (0 <= n_rows < 1 << 63 and 0 <= n_cols < 1 << 63):
        raise ValueError("dimensions must be nonnegative signed64 values")
    if n_rows * n_cols > 1 << 62:
        raise ValueError("int32 allocation exceeds the uint64 address domain")
    x = torch.arange(0, n_rows * n_cols, device="cuda", dtype=torch.int32).reshape([n_rows, n_cols])
    output = torch.zeros([n_rows, n_cols], device=x.device, dtype=x.dtype)
    n_elements = x.numel()
    if n_elements != 0:
        for tensor in (x, output):
            base = tensor.data_ptr()
            if base % 4 or base < 0 or base + 4 * n_elements > 1 << 64:
                raise ValueError("int32 buffer must be aligned and fit uint64 addresses")
        nested3_repaired[(1,)](x, output, x.stride(0), x.stride(1), n_elements)
    print(output)
