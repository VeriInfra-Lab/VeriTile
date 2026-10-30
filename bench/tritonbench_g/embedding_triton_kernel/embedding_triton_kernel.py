"""Contiguous FP32 embedding with int32 IDs; updates out and returns None."""
import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel_repaired(
    weight, input_ids, out, vob_start_id: tl.int64, vob_end_id: tl.int64,
    stride_weight_seq: tl.int64, stride_out_seq: tl.int64, n_ctx: tl.int64,
    hiden_size: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_NN: tl.constexpr,
):
    start_n = tl.program_id(0).to(tl.int64) * BLOCK_N
    offs_nn = start_n + tl.arange(0, BLOCK_NN).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)
    for start_nn in range(0, BLOCK_N, BLOCK_NN):
        start_nn = tl.multiple_of(start_nn, BLOCK_NN)
        offs_seq = start_nn + offs_nn
        n_ctx_mask = offs_seq < n_ctx
        token_ids = tl.load(input_ids + offs_seq, mask=n_ctx_mask, other=0).to(tl.int64)
        id_mask = n_ctx_mask & (token_ids >= vob_start_id) & (token_ids < vob_end_id)
        token_ids = tl.where(id_mask, token_ids, vob_start_id) - vob_start_id
        dim_mask = offs_d < hiden_size
        load_mask = id_mask[:, None] & dim_mask[None, :]
        store_mask = n_ctx_mask[:, None] & dim_mask[None, :]
        vecs = tl.load(weight + token_ids[:, None] * stride_weight_seq + offs_d[None, :],
                       mask=load_mask, other=0.0)
        output_rows = tl.where(n_ctx_mask, offs_seq, 0)
        tl.store(out + output_rows[:, None] * stride_out_seq + offs_d[None, :],
                 vecs, mask=store_mask)


@triton.jit
def embedding_kernel_index_commuted(
    weight, input_ids, out, vob_start_id: tl.int64, vob_end_id: tl.int64,
    stride_weight_seq: tl.int64, stride_out_seq: tl.int64, n_ctx: tl.int64,
    hiden_size: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_NN: tl.constexpr,
):
    start_n = BLOCK_N * tl.program_id(0).to(tl.int64)
    offs_nn = start_n + tl.arange(0, BLOCK_NN).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)
    for start_nn in range(0, BLOCK_N, BLOCK_NN):
        start_nn = tl.multiple_of(start_nn, BLOCK_NN)
        offs_seq = start_nn + offs_nn
        n_ctx_mask = offs_seq < n_ctx
        token_ids = tl.load(input_ids + offs_seq, mask=n_ctx_mask, other=0).to(tl.int64)
        id_mask = n_ctx_mask & (token_ids >= vob_start_id) & (token_ids < vob_end_id)
        token_ids = tl.where(id_mask, token_ids, vob_start_id) - vob_start_id
        dim_mask = offs_d < hiden_size
        load_mask = id_mask[:, None] & dim_mask[None, :]
        store_mask = n_ctx_mask[:, None] & dim_mask[None, :]
        vecs = tl.load(weight + token_ids[:, None] * stride_weight_seq + offs_d[None, :],
                       mask=load_mask, other=0.0)
        output_rows = tl.where(n_ctx_mask, offs_seq, 0)
        tl.store(out + output_rows[:, None] * stride_out_seq + offs_d[None, :],
                 vecs, mask=store_mask)


def _disjoint(a, b):
    return (a.numel() == 0 or b.numel() == 0
            or a.data_ptr() + 4 * a.numel() <= b.data_ptr()
            or b.data_ptr() + 4 * b.numel() <= a.data_ptr())


@torch.no_grad()
def embedding_repaired(input_ids, weight, vob_start_id, vob_end_id, out):
    tensors = (input_ids, weight, out)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("input_ids, weight and out must be tensors")
    if type(vob_start_id) is not int or type(vob_end_id) is not int:
        raise TypeError("vocabulary endpoints must be Python integers")
    if not all(t.layout == torch.strided for t in tensors):
        raise ValueError("strided tensor layout required")
    if input_ids.dtype != torch.int32 or weight.dtype != torch.float32 or out.dtype != torch.float32:
        raise TypeError("int32 IDs and FP32 weight/out required")
    if not all(t.is_cuda for t in tensors) or not (input_ids.device == weight.device == out.device):
        raise ValueError("all tensors must use the same CUDA device")
    if input_ids.ndim != 1 or weight.ndim != 2 or out.ndim != 2:
        raise ValueError("expected rank-one IDs and rank-two weight/out")
    n, hidden = input_ids.shape[0], weight.shape[1]
    if tuple(out.shape) != (n, hidden):
        raise ValueError("out shape must be (number of IDs, embedding width)")
    if not all(t.is_contiguous() and not t.is_neg() for t in tensors):
        raise ValueError("contiguous tensors without negative view bits required")
    if not all(t.numel() < 2**61 for t in tensors) or triton.cdiv(n, 64) >= 2**31 or hidden > 2**20:
        raise ValueError("tensor extent or launch dimensions exceed supported limits")
    if any(t.numel() and t.data_ptr() % 4 for t in tensors):
        raise ValueError("nonempty tensor pointers must be aligned to four bytes")
    if not _disjoint(out, input_ids) or not _disjoint(out, weight):
        raise ValueError("out must not overlap either input")
    # Intersect the interval with the int32 ID range without truncating endpoints.
    upper = min(vob_end_id, 2**31)
    intersects = max(vob_start_id, -(2**31)) < upper
    if hidden and intersects and upper - vob_start_id > weight.shape[0]:
        raise ValueError("weight must cover every selectable vocabulary row")
    if n == 0 or hidden == 0:
        return None
    first, last = (vob_start_id, upper) if intersects else (0, 0)
    with torch.cuda.device(out.device):
        embedding_kernel_repaired[(triton.cdiv(n, 64), 1, 1)](
            weight, input_ids, out, first, last, hidden, hidden, n,
            hiden_size=hidden, BLOCK_DMODEL=triton.next_power_of_2(hidden),
            BLOCK_N=64, BLOCK_NN=1, num_warps=1, num_stages=1,
        )
    return None
