"""Accumulate selected FP32 rows in ascending input order, in place."""
import torch
import triton
import triton.language as tl


@triton.jit
def index_select_cat_bwd_repaired_kernel(grad_source_ptr, grad_output_ptr,
                                        count, stride, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK).to(tl.int64)
    active = lane < count
    safe = tl.where(active, lane, 0)
    current = tl.load(grad_source_ptr + safe * stride, mask=active, other=0).to(tl.float32)
    incoming = tl.load(grad_output_ptr + safe, mask=active, other=0).to(tl.float32)
    result = current + incoming
    tl.store(grad_source_ptr + safe * stride, result, mask=active)


def index_select_cat_bwd_repaired(grad_source: torch.Tensor, index: torch.Tensor,
                                 grad_output: torch.Tensor):
    """Add each grad_output row to the selected grad_source row; return None.

    Payloads are direct, native contiguous FP32 tensors and indices are
    contiguous int64 on the same CUDA device. All validation precedes writes.
    Duplicate indices use ascending input order on the owning CUDA stream.
    """
    tensors = (grad_source, index, grad_output)
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("all arguments must be tensors")
    if any(t.layout != torch.strided for t in tensors):
        raise ValueError("all tensors must have strided layout")
    if grad_source.ndim != 2 or grad_output.ndim != 2 or index.ndim != 1:
        raise ValueError("gradients must be matrices and indices a vector")
    if grad_source.dtype != torch.float32 or grad_output.dtype != torch.float32 or index.dtype != torch.int64:
        raise ValueError("gradients must be FP32 and indices int64")
    if not grad_source.is_cuda or any(t.device != grad_source.device for t in tensors):
        raise ValueError("all tensors must share one CUDA device")
    if any(not t.is_contiguous() or t.is_neg() or t.is_conj() for t in tensors):
        raise ValueError("all tensors must be native contiguous views")
    rows, cols = grad_source.shape
    count, incoming_cols = grad_output.shape
    if rows < count or cols != incoming_cols or index.numel() != count:
        raise ValueError("require source rows >= incoming rows, equal widths, and one index per incoming row")
    if grad_source.stride() != grad_output.stride():
        raise ValueError("gradient stride tuples must match")
    if any(t.numel() * t.element_size() >= 2**63 for t in tensors):
        raise ValueError("tensor byte extents must fit signed int64")
    if any(t.numel() and t.data_ptr() % t.element_size() for t in tensors):
        raise ValueError("nonempty tensor pointers must have natural alignment")
    def separate(a, b):
        sa, sb = a.untyped_storage(), b.untyped_storage()
        return (sa.nbytes() == 0 or sb.nbytes() == 0 or sa.data_ptr() + sa.nbytes() <= sb.data_ptr()
                or sb.data_ptr() + sb.nbytes() <= sa.data_ptr())
    if not separate(grad_source, grad_output) or not separate(grad_source, index):
        raise ValueError("grad_source backing storage must be disjoint from read roles")
    if count == 0 or cols == 0:
        return None
    indices = index.cpu().tolist()
    if any(i < 0 or i >= rows for i in indices):
        raise IndexError("every index must select a grad_source row")
    # Views preserve the original storage and offset. Canonical physical
    # strides also cover irrelevant singleton strides without copying data.
    source_words = grad_source.as_strided((grad_source.numel(),), (1,))
    incoming_words = grad_output.as_strided((grad_output.numel(),), (1,))
    with torch.cuda.device(grad_source.device):
        for i, row in enumerate(indices):
            for start in range(0, cols, 512):
                size = min(512, cols - start)
                index_select_cat_bwd_repaired_kernel[(1,)](
                    source_words[row * cols + start:row * cols + start + size],
                    incoming_words[i * cols + start:i * cols + start + size],
                    size, 1, BLOCK=512)
    return None
