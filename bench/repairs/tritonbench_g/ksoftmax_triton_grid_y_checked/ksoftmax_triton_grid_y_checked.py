
import torch
import triton
import triton.language as tl

def get_depth(K):
    return triton.next_power_of_2(K)

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["K"],
)
@triton.heuristics({'DEPTH': lambda nargs: get_depth(nargs['K'])})
@triton.heuristics({'IS_FP16': lambda nargs: nargs['Y'].dtype == torch.float16})
@triton.jit
def _softmax(
    Y, X, M,
    stride_ym: tl.int64, stride_yn: tl.int64,
    stride_xm: tl.int64, stride_xn: tl.int64,
    stride_m: tl.int64,
    K: tl.int64,
    LOG: tl.constexpr,
    MASK_TYPE: tl.constexpr,
    CAUSAL: tl.constexpr,
    DEPTH: tl.constexpr,
    IS_FP16: tl.constexpr,
):
    """
    Fused softmax kernel over a 3d tensor.
    The softmax is applied over the last dimension, equivalent to torch.softmax(tensor, dim=-1)
    """
    m = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1).to(tl.int64)
    k = tl.arange(0, DEPTH).to(tl.int64)
    x_ptrs = X + m * stride_xm + n * stride_xn + k
    io_mask = k < K
    if CAUSAL:
        io_mask = io_mask & (k <= n)
    x = tl.load(x_ptrs, mask=io_mask, other=float("-inf"))
    if CAUSAL:
        off = float("-inf")
        off = off.to(x.dtype)
        x = tl.where(k > n, off, x)
    if MASK_TYPE is not None:
        if MASK_TYPE == 'qk':
            mask_ptrs = M + n * stride_m + k
        elif MASK_TYPE == 'bk':
            mask_ptrs = M + m * stride_m + k
        add_mask = tl.load(mask_ptrs, io_mask, other=float("-inf"))
        x += add_mask
    z = x - tl.max(x, axis=0)
    if IS_FP16:
        z = z.to(tl.float32)
    num = tl.exp(z)
    denom = tl.sum(num, axis=0)
    if LOG:
        y = z - tl.log(denom)
    else:
        y = num / denom
    y_ptrs = Y + m * stride_ym + n * stride_yn + k
    tl.store(y_ptrs, y, mask=k < K)

def get_depth(K):
    return triton.next_power_of_2(K)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["K"],
)
@triton.heuristics({'DEPTH': lambda nargs: get_depth(nargs['K'])})
@triton.heuristics({'IS_FP16': lambda nargs: nargs['GradIn'].dtype == torch.float16})
@triton.jit
def _softmax_backward(
    GradIn, GradOut, Out,
    stride_bm: tl.int64, stride_bn: tl.int64,
    stride_gm: tl.int64, stride_gn: tl.int64,
    stride_om: tl.int64, stride_on: tl.int64,
    K: tl.int64,
    LOG: tl.constexpr,
    CAUSAL: tl.constexpr,
    DEPTH: tl.constexpr,
    IS_FP16: tl.constexpr,
):
    """
    Compute the softmax gradients.
    """
    m = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1).to(tl.int64)
    k = tl.arange(0, DEPTH).to(tl.int64)
    grad_out_ptrs = GradOut + m * stride_gm + n * stride_gn + k
    out_ptrs = Out + m * stride_om + n * stride_on + k
    io_mask = k < K
    if CAUSAL:
        io_mask = io_mask & (k <= n)
    g = tl.load(grad_out_ptrs, mask=io_mask, other=float(0))
    o = tl.load(out_ptrs, mask=io_mask, other=float(0))
    if CAUSAL:
        zero = float(0)
        zero = zero.to(g.dtype)
        g = tl.where(k > n, zero, g)
        o = tl.where(k > n, zero, o)
    if LOG:
        s = tl.sum(g, 0)
        if IS_FP16:
            o = o.to(tl.float32)
        grad_in = g - tl.exp(o) * s
    else:
        s = tl.sum(g * o, 0)
        grad_in = o * (g - s)
    if CAUSAL:
        grad_in = tl.where(k <= n, grad_in, 0.)
    grad_in_ptrs = GradIn + m * stride_bm + n * stride_bn + k
    tl.store(grad_in_ptrs, grad_in, mask=k < K)


def _tensor(t):
    if not isinstance(t, torch.Tensor):
        raise TypeError("all active tensor roles must be Tensors")
    if t.layout != torch.strided or t.dtype != torch.float32 or not t.is_cuda:
        raise ValueError("active tensors must be strided FP32 CUDA tensors")
    if not t.is_contiguous() or t.is_neg():
        raise ValueError("active tensors must be contiguous without negative view bits")


def _ranges(destination, inputs):
    roles = (destination, *inputs)
    pointers = [t.data_ptr() for t in roles]
    if any(p % 4 for p in pointers):
        raise ValueError("FP32 pointers must be 4-byte aligned")
    base, end = pointers[0], pointers[0] + 4 * destination.numel()
    for t, p in zip(inputs, pointers[1:]):
        if not (end <= p or p + 4 * t.numel() <= base):
            raise ValueError("destination must not overlap any read tensor")


def _extent(x):
    if x.shape[0] >= 2**31 or x.shape[1] >= 2**31 or x.shape[2] > 2**30 or x.numel() >= 2**61:
        raise ValueError("shape exceeds the source integer index domain")
    if x.shape[1] > 65535:
        raise ValueError("query rows exceed the CUDA grid y limit 65535")


def softmax(Y, X, M=None, log=False, mask_type=None, causal=False):
    if type(log) is not bool or type(causal) is not bool:
        raise TypeError("log and causal must be bool")
    if mask_type not in (None, 'qk', 'bk'):
        raise ValueError("mask_type must be None, 'qk', or 'bk'")
    _tensor(Y)
    _tensor(X)
    if X.ndim != 3 or Y.shape != X.shape:
        raise ValueError("X and Y must have the same rank-three shape")
    if Y.device != X.device:
        raise ValueError("all active tensor roles must share one device")
    inputs = (X,)
    if mask_type is not None:
        _tensor(M)
        rows = X.shape[1] if mask_type == 'qk' else X.shape[0]
        if M.shape != (rows, X.shape[2]) or M.device != X.device:
            raise ValueError("mask shape/device must match its qk or bk role")
        inputs = (X, M)
    if X.numel() == 0:
        return
    _extent(X)
    _ranges(Y, inputs)
    B, Q, K = X.shape
    with torch.cuda.device(X.device):
        _softmax[(B, Q)](Y, X, M if mask_type is not None else X,
            Q*K, K, Q*K, K, K if mask_type is not None else 0, K,
            LOG=log, MASK_TYPE=mask_type, CAUSAL=causal, enable_fp_fusion=True)


def softmax_backward(GradIn, GradOut, Out, log=False, causal=False):
    if type(log) is not bool or type(causal) is not bool:
        raise TypeError("log and causal must be bool")
    for t in (GradIn, GradOut, Out):
        _tensor(t)
    if Out.ndim != 3 or GradIn.shape != Out.shape or GradOut.shape != Out.shape:
        raise ValueError("all gradient tensors must have the same rank-three shape")
    if GradIn.device != Out.device or GradOut.device != Out.device:
        raise ValueError("all active tensor roles must share one device")
    if Out.numel() == 0:
        return
    _extent(Out)
    _ranges(GradIn, (GradOut, Out))
    B, Q, K = Out.shape
    with torch.cuda.device(Out.device):
        _softmax_backward[(B, Q)](GradIn, GradOut, Out,
            Q*K, K, Q*K, K, Q*K, K, K, LOG=log, CAUSAL=causal,
            enable_fp_fusion=True)
