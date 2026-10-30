import torch
import triton
import triton.language as tl

@triton.jit
def mean_dim_kernel(X, Mean, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Map the program id to the row of X it should compute.
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None] 
    X = X + pid * N
    Mean = Mean + pid
    row_mask = pid < M

    # Compute mean
    _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=1) / N
    mean = mean[:, None]
    tl.store(Mean, mean, row_mask)

# Existing proof variant: commute only exact integer index multiplication.
@triton.jit
def mean_dim_kernel_index_commuted(X, Mean, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Map the program id to the row of X it should compute.
    pid = BLOCK_M * tl.program_id(0) + tl.arange(0, BLOCK_M)[:, None] 
    X = X + pid * N
    Mean = Mean + pid
    row_mask = pid < M

    # Compute mean
    _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=1) / N
    mean = mean[:, None]
    tl.store(Mean, mean, row_mask)

def dim_compress(inp: torch.Tensor, dims):
    if isinstance(dims, int):
        dims = [dims]
    dim = inp.ndim
    stride = inp.stride()
    batch_dim = [i for i in range(dim) if i not in dims]
    sorted_reduction_dim = sorted(dims, key=lambda x: stride[x], reverse=True)
    order = batch_dim + sorted_reduction_dim
    return inp.permute(order).contiguous()

def mean_dim(x, dim, keepdim=False, *, dtype=None):
  if not isinstance(x, torch.Tensor):
    raise TypeError("x must be a Tensor")
  if x.layout != torch.strided:
    raise ValueError("x must use strided layout")
  if x.dtype != torch.float32:
    raise TypeError("x must be FP32")
  if dtype is not None and dtype != torch.float32:
    raise TypeError("dtype must be None or FP32")
  if not x.is_cuda:
    raise ValueError("x must be CUDA")
  if x.is_neg():
    raise ValueError("x must have no unresolved negative view bit")
  if x.numel() > 0 and x.data_ptr() % 4 != 0:
    raise ValueError("x must have a 4-byte-aligned data pointer")
  if isinstance(dim, int):
    checked_dims = [dim]
  elif isinstance(dim, list) and all(isinstance(d, int) for d in dim):
    checked_dims = dim
  else:
    raise TypeError("dim must be an int or list of ints")
  if x.ndim == 0 and checked_dims:
    raise ValueError("a scalar supports only an empty dimension list")
  checked_dims = [d % x.ndim for d in checked_dims]
  if len(set(checked_dims)) != len(checked_dims):
    raise ValueError("normalized reduction dimensions must be distinct")
  checked_n = 1
  for d in checked_dims:
    checked_n *= x.shape[d]
  if checked_n == 0:
    raise ValueError("selected reduction extent must be positive")
  checked_m = x.numel() // checked_n
  padded_m = ((checked_m + 7) // 8) * 8
  padded_n = ((checked_n + 7) // 8) * 8
  if padded_m >= 2**31 or padded_n >= 2**31:
    raise ValueError("padded dimensions must fit signed int32")
  if checked_m > 0 and (padded_m - 1) * checked_n >= 2**31:
    raise ValueError("padded row products must fit signed int32")
  if dtype is None:
    dtype = x.dtype
  
  shape = list(x.shape)
  if isinstance(dim, int):
     dim = [dim]
  dim = [d % x.ndim for d in dim]
  x = dim_compress(x, dim)
  N = 1
  for i in dim:
    N *= shape[i]
    shape[i] = 1
  M = x.numel() // N
  out = torch.empty(shape, dtype=dtype, device=x.device)
  grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

  with torch.cuda.device(x.device):
    mean_dim_kernel[grid](x, out, M, N, BLOCK_M=8, BLOCK_N=8)
  if not keepdim:
    out = out.squeeze(dim)
  return out



