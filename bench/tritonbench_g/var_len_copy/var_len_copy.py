"""Direct contiguous FP32 segment copy with signed-int32 metadata.

The repaired kernel uses i+lane tail masks and widens indexing before arithmetic.
The original output is mutated; all read buffers must be disjoint from it.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def var_len_copy_kernel_triton_repaired(old_a_start, old_a_len, old_a_location, new_a_start, new_a_location,
                               BLOCK_SIZE: tl.constexpr):
    a_id = tl.cast(tl.program_id(0), tl.int64)
    length = tl.cast(tl.load(old_a_len + a_id), tl.int64)
    old_start = tl.cast(tl.load(old_a_start + a_id), tl.int64)
    new_start = tl.cast(tl.load(new_a_start + a_id), tl.int64)
    old_offset = tl.cast(tl.arange(0, BLOCK_SIZE), tl.int64)
    new_offset = tl.cast(tl.arange(0, BLOCK_SIZE), tl.int64)
    for i in range(0, length, BLOCK_SIZE):
        v = tl.load(old_a_location + old_start + i + old_offset, mask=i + old_offset < length)
        tl.store(new_a_location + new_start + i + new_offset, v, mask=i + new_offset < length)


@triton.jit
def var_len_copy_kernel_triton_commuted(old_a_start, old_a_len, old_a_location, new_a_start, new_a_location,
                               BLOCK_SIZE: tl.constexpr):
    a_id = tl.cast(tl.program_id(0), tl.int64)
    length = tl.cast(tl.load(old_a_len + a_id), tl.int64)
    old_start = tl.cast(tl.load(old_a_start + a_id), tl.int64)
    new_start = tl.cast(tl.load(new_a_start + a_id), tl.int64)
    old_offset = tl.cast(tl.arange(0, BLOCK_SIZE), tl.int64)
    new_offset = tl.cast(tl.arange(0, BLOCK_SIZE), tl.int64)
    for i in range(0, length, BLOCK_SIZE):
        v = tl.load(old_a_location + i + old_start + old_offset, mask=i + old_offset < length)
        tl.store(new_a_location + new_start + i + new_offset, v, mask=i + new_offset < length)


def launch_var_len_copy_triton_repaired(old_a_start, old_a_len, old_location, new_a_start, new_a_location):
    tensors = (old_a_start, old_a_len, old_location, new_a_start, new_a_location)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("all arguments must be tensors")
    if any(t.layout != torch.strided for t in tensors):
        raise TypeError("strided tensors required")
    if any(t.ndim != 1 for t in tensors):
        raise ValueError("all arguments must be vectors")
    if (old_a_start.dtype != torch.int32 or old_a_len.dtype != torch.int32 or
            new_a_start.dtype != torch.int32 or old_location.dtype != torch.float32 or
            new_a_location.dtype != torch.float32):
        raise TypeError("FP32 payloads and int32 metadata required")
    if any(not t.is_cuda for t in tensors):
        raise ValueError("CUDA tensors required")
    if any(t.device != old_location.device for t in tensors):
        raise ValueError("all tensors must use one device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("contiguous tensors required")
    if any(t.is_neg() for t in tensors):
        raise ValueError("unresolved negative views are unsupported")
    if old_a_start.shape != old_a_len.shape or old_a_start.shape != new_a_start.shape:
        raise ValueError("metadata vectors must have equal shapes")
    count = old_a_start.numel()
    if count >= (1 << 31):
        raise ValueError("metadata count must be below 2**31")
    if any(t.numel() > 0 and t.data_ptr() % 4 != 0 for t in tensors):
        raise ValueError("nonempty data pointers must be 4-byte aligned")
    out_begin = new_a_location.data_ptr()
    out_end = out_begin + 4 * new_a_location.numel()
    if any(not (out_end <= t.data_ptr() or t.data_ptr() + 4 * t.numel() <= out_begin)
           for t in tensors[:-1]):
        raise ValueError("output must be disjoint from every read buffer")

    # Only metadata is read back for admission; payload tensors are passed directly.
    old_starts = old_a_start.cpu().tolist()
    lengths = old_a_len.cpu().tolist()
    new_starts = new_a_start.cpu().tolist()
    if any(length > 0 and (start < 0 or start + length > old_location.numel())
           for start, length in zip(old_starts, lengths)):
        raise ValueError("source segment out of bounds")
    if any(length > 0 and (start < 0 or start + length > new_a_location.numel())
           for start, length in zip(new_starts, lengths)):
        raise ValueError("output segment out of bounds")
    if any(lengths[p] > 0 and lengths[q] > 0 and
           not (new_starts[p] + lengths[p] <= new_starts[q] or
                new_starts[q] + lengths[q] <= new_starts[p])
           for p in range(count) for q in range(p + 1, count)):
        raise ValueError("destination segments must be disjoint")
    if count == 0:
        return None
    with torch.cuda.device(old_location.device):
        var_len_copy_kernel_triton_repaired[(count,)](
            old_a_start, old_a_len, old_location, new_a_start, new_a_location, 256)
    return None
