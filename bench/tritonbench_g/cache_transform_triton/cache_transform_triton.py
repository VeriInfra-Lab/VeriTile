"""Direct FP32 cache gather; prompt concatenation and decode length-minus-one."""
import torch
import triton
import triton.language as tl


@triton.jit
def gather_cache_bytes(Cache, Rows, Output, COUNT: tl.constexpr,
                       ROW_BYTES: tl.constexpr, START: tl.constexpr,
                       BLOCK: tl.constexpr):
    idx = tl.program_id(0).to(tl.uint64) * BLOCK + tl.arange(0, BLOCK).to(tl.uint64) + START
    active = idx < COUNT
    safe = tl.where(active, idx, 0)
    row = tl.load(Rows + safe // ROW_BYTES).to(tl.uint64)
    offset = row * ROW_BYTES + safe % ROW_BYTES
    value = tl.load(Cache + offset, mask=active, other=0)
    tl.store(Output + safe, value, mask=active)


def _gather(cache, rows):
    output = torch.empty((len(rows), cache.shape[1]), dtype=torch.float32, device=cache.device)
    count = output.numel()*4
    if count:
        table = torch.tensor(rows, dtype=torch.int64, device=cache.device)
        # A flat physical view also handles irrelevant singleton-dimension strides.
        cache_bytes = cache.as_strided((cache.numel(),), (1,)).view(torch.uint8)
        output_bytes = output.view(torch.uint8).reshape(-1)
        gather_cache_bytes[(triton.cdiv(count,256),)](
            cache_bytes, table, output_bytes, COUNT=count,
            ROW_BYTES=cache.shape[1]*4, START=0, BLOCK=256, num_warps=4)
    return output


def get_xine_cache(lengths: torch.Tensor, cos_cache: torch.Tensor,
                   sin_cache: torch.Tensor, is_prompts: bool = False):
    """Return two fresh contiguous FP32 CUDA tensors; caller inputs are read-only.

    Lengths use int32 on the same CUDA device. Caches are rank-two native
    contiguous FP32 tensors, with equal hidden width and independent row counts.
    Prompt lengths may be zero; decode lengths must be positive. Empty work
    returns correctly shaped empty tensors without launching a kernel.
    """
    if type(is_prompts) is not bool:
        raise TypeError("is_prompts must be bool")
    if any(not isinstance(t,torch.Tensor) for t in (lengths,cos_cache,sin_cache)):
        raise TypeError("lengths and both caches must be tensors")
    if any(t.layout != torch.strided for t in (lengths,cos_cache,sin_cache)):
        raise ValueError("inputs must have strided layout")
    if lengths.ndim != 1 or lengths.dtype != torch.int32:
        raise ValueError("lengths must be rank-one int32")
    if not cos_cache.is_cuda or any(t.device != cos_cache.device for t in (lengths,sin_cache)):
        raise ValueError("inputs must share one CUDA device")
    for cache in (cos_cache,sin_cache):
        if cache.ndim != 2 or cache.dtype != torch.float32:
            raise ValueError("caches must be rank-two FP32")
        if not cache.is_contiguous() or cache.is_neg() or cache.is_conj():
            raise ValueError("caches must be native contiguous tensors")
        if cache.numel()*4 >= 2**63:
            raise ValueError("cache byte extents must fit signed int64")
    if cos_cache.shape[1] != sin_cache.shape[1]:
        raise ValueError("cache hidden widths differ")
    values = lengths.cpu().tolist()
    minimum = 0 if is_prompts else 1
    if any(v < minimum or v > cos_cache.shape[0] or v > sin_cache.shape[0] for v in values):
        raise ValueError("lengths are outside the cache row bounds")
    row_count = sum(values) if is_prompts else len(values)
    count = row_count*cos_cache.shape[1]*4
    if count >= 2**63 or triton.cdiv(count,256) >= 2**31:
        raise ValueError("output byte extent or launch grid is too large")
    if count and row_count*8 >= 2**63:
        raise ValueError("row-table byte extent must fit signed int64")
    rows = [r for v in values for r in range(v)] if is_prompts else [v-1 for v in values]
    with torch.cuda.device(cos_cache.device):
        return _gather(cos_cache,rows), _gather(sin_cache,rows)
