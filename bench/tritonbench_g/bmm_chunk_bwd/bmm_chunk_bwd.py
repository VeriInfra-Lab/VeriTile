import math
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_CS': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_CS': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_CS': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=2),
    ],
    key=['chunk_size', 'K'],
)
@triton.jit
def _bmm_chunk_bwd_kernel(
    a_ptr, dout_ptr, db_ptr, res_ptr,
    seqlen, chunk_size, K, ngroups,
    stride_a_batch, stride_a_seqlen, stride_a_head, stride_ak,
    stride_dout_batch, stride_dout_chunk, stride_dout_head, stride_dout_csize_m, stride_dout_csize_n,
    stride_db_batch, stride_db_seqlen, stride_db_head, stride_db_k,
    stride_res_batch, stride_res_seqlen, stride_res_head, stride_res_k,
    dot_dtype: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_CS: tl.constexpr,
):
    seqlen = tl.cast(seqlen, tl.int64)
    chunk_size = tl.cast(chunk_size, tl.int64)
    K = tl.cast(K, tl.int64)
    ngroups = tl.cast(ngroups, tl.int64)
    stride_a_batch = tl.cast(stride_a_batch, tl.int64)
    stride_a_seqlen = tl.cast(stride_a_seqlen, tl.int64)
    stride_a_head = tl.cast(stride_a_head, tl.int64)
    stride_ak = tl.cast(stride_ak, tl.int64)
    stride_dout_batch = tl.cast(stride_dout_batch, tl.int64)
    stride_dout_chunk = tl.cast(stride_dout_chunk, tl.int64)
    stride_dout_head = tl.cast(stride_dout_head, tl.int64)
    stride_dout_csize_m = tl.cast(stride_dout_csize_m, tl.int64)
    stride_dout_csize_n = tl.cast(stride_dout_csize_n, tl.int64)
    stride_db_batch = tl.cast(stride_db_batch, tl.int64)
    stride_db_seqlen = tl.cast(stride_db_seqlen, tl.int64)
    stride_db_head = tl.cast(stride_db_head, tl.int64)
    stride_db_k = tl.cast(stride_db_k, tl.int64)
    stride_res_batch = tl.cast(stride_res_batch, tl.int64)
    stride_res_seqlen = tl.cast(stride_res_seqlen, tl.int64)
    stride_res_head = tl.cast(stride_res_head, tl.int64)
    stride_res_k = tl.cast(stride_res_k, tl.int64)
    pid_b = tl.program_id(axis=1).to(tl.int64)
    pid_ch = tl.program_id(axis=2).to(tl.int64)
    pid_c = pid_ch // ngroups
    pid_h = pid_ch - pid_c * ngroups
    num_pid_n = tl.cdiv(K, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0).to(tl.int64) // num_pid_n
    pid_n = tl.program_id(axis=0).to(tl.int64) % num_pid_n

    a_ptr += pid_b * stride_a_batch + pid_c * chunk_size * stride_a_seqlen + pid_h * stride_a_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * stride_dout_chunk + pid_h * stride_dout_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cs = tl.arange(0, BLOCK_SIZE_CS).to(tl.int64)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_csize_n + offs_cs[None, :] * stride_dout_csize_m)
    a_ptrs = a_ptr + (offs_cs[:, None] * stride_a_seqlen + offs_n[None, :] * stride_ak)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for cs in range(0, tl.cdiv(chunk_size_limit, BLOCK_SIZE_CS)):
        dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_cs[None, :] < chunk_size_limit - cs * BLOCK_SIZE_CS), other=0.0).to(dot_dtype)
        a = tl.load(a_ptrs, mask=(offs_cs[:, None] < chunk_size_limit - cs * BLOCK_SIZE_CS) & (offs_n[None, :] < K), other=0.0).to(dot_dtype)
        acc += tl.dot(dout, a)
        dout_ptrs += BLOCK_SIZE_CS * stride_dout_csize_m
        a_ptrs += BLOCK_SIZE_CS * stride_a_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if HAS_RESIDUAL:
        res_ptr += pid_b * stride_res_batch + pid_c * chunk_size * stride_res_seqlen + pid_h * stride_res_head
        res_ptrs = res_ptr + (offs_m[:, None] * stride_res_seqlen + offs_n[None, :] * stride_res_k)
        res = tl.load(res_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < K)).to(tl.float32)
        acc += res
    db = acc.to(db_ptr.dtype.element_ty)

    db_ptr += pid_b * stride_db_batch + pid_c * chunk_size * stride_db_seqlen + pid_h * stride_db_head
    db_ptrs = db_ptr + (offs_m[:, None] * stride_db_seqlen + offs_n[None, :] * stride_db_k)
    tl.store(db_ptrs, db, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < K))

def _bmm_chunk_bwd(a, dout, residual=None, out=None):
    tensors=(a,dout)+(() if residual is None else (residual,))+(() if out is None else (out,))
    if not all(isinstance(t,torch.Tensor) for t in tensors):
        raise TypeError("Tensor arguments required")
    if any(t.layout != torch.strided for t in tensors):
        raise TypeError("strided tensors required")
    if any(t.dtype != torch.float32 for t in tensors):
        raise TypeError("FP32 tensors required")
    if any(not t.is_cuda for t in tensors):
        raise ValueError("CUDA tensors required")
    if any(t.device != a.device for t in tensors):
        raise ValueError("one CUDA device required")
    if a.ndim not in (3,4) or dout.ndim != a.ndim+1:
        raise ValueError("A rank 3/4 and Dout rank 4/5 required")
    has_groups=a.ndim==4
    batch,seqlen=a.shape[:2]
    ngroups,k=(a.shape[2],a.shape[3]) if has_groups else (1,a.shape[2])
    nchunks,chunk_size=dout.shape[1],dout.shape[-1]
    if chunk_size<=0:
        raise ValueError("positive chunk size required")
    expected=((batch,nchunks,ngroups,chunk_size,chunk_size) if has_groups else
              (batch,nchunks,chunk_size,chunk_size))
    if dout.shape != expected or nchunks != (seqlen+chunk_size-1)//chunk_size:
        raise ValueError("Dout must contain the full square chunk grid")
    if residual is not None and residual.shape != a.shape:
        raise ValueError("residual shape must match A")
    if out is not None and out.shape != a.shape:
        raise ValueError("output shape must match A")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("contiguous tensors required")
    strides=(seqlen*ngroups*k,ngroups*k,k if has_groups else 0,1)
    dstrides=(nchunks*ngroups*chunk_size*chunk_size,ngroups*chunk_size*chunk_size,
              chunk_size*chunk_size if has_groups else 0,chunk_size,1)
    tensor_roles=(a,)+(() if residual is None else (residual,))+(() if out is None else (out,))
    if any(t.stride(-1)!=1 or (t.numel() and
               (t.stride(0),t.stride(1),t.stride(2) if has_groups else 0,t.stride(-1))!=strides)
           for t in tensor_roles) or dout.stride(-1)!=1 or (dout.numel() and
               (dout.stride(0),dout.stride(1),dout.stride(2) if has_groups else 0,dout.stride(-2),dout.stride(-1))!=dstrides):
        raise ValueError("literal contiguous source strides required")
    if any(t.is_neg() for t in tensors):
        raise ValueError("unresolved negative views unsupported")
    if any(4*t.numel()>=(1<<63) for t in tensors):
        raise ValueError("byte extents must fit signed int64")
    if a.numel() and (batch>=(1<<16) or nchunks*ngroups>=(1<<16) or
          any(triton.cdiv(chunk_size,m)*triton.cdiv(k,n)>=(1<<31)
              for m,n in ((128,256),(64,256),(128,128),(128,64),(64,128),(128,32),(64,32),(32,64),(64,64)))):
        raise ValueError("launch dimensions exceed CUDA limits")
    if any(t.numel() and t.data_ptr()%4 for t in tensors):
        raise ValueError("aligned FP32 data required")
    if any(not(x.data_ptr()+4*x.numel()<=y.data_ptr() or y.data_ptr()+4*y.numel()<=x.data_ptr())
           for i,x in enumerate(tensors) for y in tensors[i+1:]):
        raise ValueError("tensor data spans must be disjoint")
    # Original intentional layout operations remain at their source positions.
    if a.stride(-1) != 1 and a.stride(-2) != 1:
        a = a.contiguous()
    if dout.stride(-1) != 1 and dout.stride(-2) != 1:
        dout = dout.contiguous()
    if residual is not None:
        if residual.stride(-1) != 1 and residual.stride(1) != 1:
            residual = residual.contiguous()
    if out is None:
        out = torch.empty_like(a)
    if a.numel()==0:
        return out
    grid=lambda meta:(triton.cdiv(chunk_size,meta['BLOCK_SIZE_M'])*triton.cdiv(k,meta['BLOCK_SIZE_N']),batch,nchunks*ngroups)
    residual_strides=((residual.stride(0),residual.stride(1),residual.stride(2) if has_groups else 0,residual.stride(-1))
                      if residual is not None else (0,0,0,0))
    with torch.cuda.device(a.device):
        _bmm_chunk_bwd_kernel[grid](
            a,dout,out,residual,seqlen,chunk_size,k,ngroups,
            a.stride(0),a.stride(1),a.stride(2) if has_groups else 0,a.stride(-1),
            dout.stride(0),dout.stride(1),dout.stride(2) if has_groups else 0,dout.stride(-2),dout.stride(-1),
            out.stride(0),out.stride(1),out.stride(2) if has_groups else 0,out.stride(-1),
            *residual_strides,tl.float32,HAS_RESIDUAL=residual is not None)
    return out
