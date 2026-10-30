"""Direct contiguous FP32 gather with int64 indices; returns the supplied output."""
import torch
import triton
import triton.language as tl


@triton.jit
def index_select_cat_fwd_kernel_repaired(output_ptr, source_ptr, index_ptr,
    num_indices, num_cols, stride0, stride1,
    BLOCK_SIZE_INDEX: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr):
    pid0 = tl.program_id(axis=0).to(tl.int64)
    pid1 = tl.program_id(axis=1).to(tl.int64)
    indices = pid0 * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX).to(tl.int64)
    rows = tl.load(index_ptr + indices, mask=indices < num_indices, other=0).to(tl.int64)
    cols = pid1 * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL).to(tl.int64)
    source_offsets = source_ptr + rows[:, None] * stride0 + cols[None, :] * stride1
    mask = (indices[:, None] < num_indices) & (cols[None, :] < num_cols)
    output = tl.load(source_offsets, mask=mask)
    output_offsets = output_ptr + indices[:, None] * stride0 + cols[None, :] * stride1
    tl.store(output_offsets, output, mask=mask)


@triton.jit
def index_select_cat_fwd_kernel_index_commuted(output_ptr, source_ptr, index_ptr,
    num_indices, num_cols, stride0, stride1,
    BLOCK_SIZE_INDEX: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr):
    pid0 = tl.program_id(axis=0).to(tl.int64)
    pid1 = tl.program_id(axis=1).to(tl.int64)
    indices = BLOCK_SIZE_INDEX * pid0 + tl.arange(0, BLOCK_SIZE_INDEX).to(tl.int64)
    rows = tl.load(index_ptr + indices, mask=indices < num_indices, other=0).to(tl.int64)
    cols = pid1 * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL).to(tl.int64)
    source_offsets = source_ptr + rows[:, None] * stride0 + cols[None, :] * stride1
    mask = (indices[:, None] < num_indices) & (cols[None, :] < num_cols)
    output = tl.load(source_offsets, mask=mask)
    output_offsets = output_ptr + indices[:, None] * stride0 + cols[None, :] * stride1
    tl.store(output_offsets, output, mask=mask)


def index_select_cat_fwd_repaired(output: torch.Tensor, source: torch.Tensor, index: torch.Tensor):
    """Write every selected row, including repeated IDs and N > M, in place.

    Source/output are contiguous FP32 CUDA matrices; IDs are contiguous int64
    on the same device. The output has shape [len(index), source.shape[1]] and
    disjoint backing storage. No payload staging or output copyback is used.
    """
    if any(not isinstance(t,torch.Tensor) for t in (output,source,index)):
        raise TypeError("all arguments must be tensors")
    if any(t.layout != torch.strided for t in (output,source,index)):
        raise ValueError("inputs must have strided layout")
    if source.ndim != 2 or output.ndim != 2 or index.ndim != 1:
        raise ValueError("source/output must be matrices and index must be a vector")
    if source.dtype != torch.float32 or output.dtype != torch.float32 or index.dtype != torch.int64:
        raise ValueError("source/output must be FP32 and index must be int64")
    if not source.is_cuda or output.device != source.device or index.device != source.device:
        raise ValueError("all tensors must share one CUDA device")
    if any(not t.is_contiguous() or t.is_neg() or t.is_conj() for t in (output,source,index)):
        raise ValueError("all tensors must be native contiguous views")
    m,d = source.shape
    n = index.numel()
    if output.shape != (n,d):
        raise ValueError("output shape must be [number of indices, source width]")
    if any(t.numel()*t.element_size() >= 2**63 for t in (output,source,index)):
        raise ValueError("tensor byte extents must fit signed int64")
    if any(t.numel() and t.data_ptr()%t.element_size() for t in (output,source,index)):
        raise ValueError("nonempty tensor pointers must have natural alignment")
    def separate(a,b):
        sa,sb = a.untyped_storage(),b.untyped_storage()
        return (sa.nbytes()==0 or sb.nbytes()==0 or sa.data_ptr()+sa.nbytes()<=sb.data_ptr()
                or sb.data_ptr()+sb.nbytes()<=sa.data_ptr())
    if not separate(output,source) or not separate(output,index):
        raise ValueError("output backing storage must be disjoint from source and index")
    if n >= 2**31 or triton.cdiv(d,512) > 65535:
        raise ValueError("launch dimensions exceed native grid bounds")
    if n == 0 or d == 0:
        return output
    if any(i < 0 or i >= m for i in index.cpu().tolist()):
        raise IndexError("every index must select a source row")
    with torch.cuda.device(source.device):
        index_select_cat_fwd_kernel_repaired[(n,triton.cdiv(d,512))](
            output,source,index,n,d,d,1,BLOCK_SIZE_INDEX=1,BLOCK_SIZE_COL=512)
    return output
