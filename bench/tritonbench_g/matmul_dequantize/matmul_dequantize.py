
import torch
import triton
import triton.language as tl

from functools import lru_cache

_I32 = 1 << 31
_U64 = 1 << 64


class _Index(int):
    """Check every node of the source's nonnegative signed-int32 expressions."""
    def __new__(cls, value):
        if not 0 <= value < _I32:
            raise ValueError("source expression exceeds nonnegative signed int32")
        return int.__new__(cls, value)
    def __add__(self, rhs): return _Index(int(self) + int(rhs))
    def __radd__(self, lhs): return _Index(int(lhs) + int(self))
    def __sub__(self, rhs): return _Index(int(self) - int(rhs))
    def __mul__(self, rhs): return _Index(int(self) * int(rhs))
    def __rmul__(self, lhs): return _Index(int(lhs) * int(self))
    def __floordiv__(self, rhs): return _Index(int(self) // int(rhs))
    def __mod__(self, rhs): return _Index(int(self) % int(rhs))


def _cdiv(x, y): return (_Index(x) + _Index(y) - 1) // y


def _dense(t, dtype, name, device):
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if (t.ndim != 2 or t.dtype != dtype or t.layout != torch.strided or
            not t.is_cuda or t.device != device or t.is_neg() or t.is_conj()):
        raise ValueError(f"{name} requires a resolved rank-two {dtype} Tensor on {device}")
    rows, cols = t.shape
    if rows <= 0 or cols <= 0 or not t.is_contiguous() or t.stride() != (cols, 1):
        raise ValueError(f"{name} requires positive shape and canonical contiguous strides")
    storage, width = t.untyped_storage(), t.element_size()
    base, size, offset, ptr = storage.data_ptr(), storage.nbytes(), t.storage_offset(), t.data_ptr()
    if (offset < 0 or base < 0 or size % width or base + size >= _U64 or
            ptr != base + width * offset or ptr % width or
            width * (offset + rows * cols) > size or ptr + width * _I32 >= _U64):
        raise ValueError(f"{name} has an invalid allocation or formed pointer span")
    return base, base + size, size // width - offset


def _apart(spans):
    for i, (lo, hi, _) in enumerate(spans):
        if any(not (hi <= x or y <= lo) for x, y, _ in spans[:i]):
            raise ValueError("all live tensor allocations must be disjoint")


@lru_cache(maxsize=128)
def _source_indices(route, M, N, K, group, capacities):
    """Literal ABI roots/accesses: source-specific guards, no launcher limit."""
    N, K, group = map(_Index, (N, K, group))
    if route in ("gptq", "s2"): M = _Index(M)
    _Index(8 * (K // 8))
    # Canonical contiguous strides admitted by _dense.
    strides = ((K, _Index(1)), (N, _Index(1)), (N, _Index(1)),
               (N // 8, _Index(1)), (N, _Index(1)), (N, _Index(1)))
    def coord(t, r, c): return _Index(r) * strides[t][0] + _Index(c) * strides[t][1]
    def access(t, offset, active):
        if active and not offset < capacities[t]:
            raise ValueError("active source access exceeds its allocation")
    if route in ('gptq', 's2'):
        for bm, bn in (((64, 256), (128, 128)) if route == 'gptq' else ((64, 128),)):
            second = route == 's2'
            nm, nn, nk = _cdiv(M, bm), _cdiv(N, bn), _cdiv(K, 32)
            ng = 8 * nn
            ai = (32 * _Index(1) if second else _Index(32)) * strides[0][1]
            bi = ((32 * _Index(1)) * strides[1][0]) // 8 if second else (32 // _Index(8)) * strides[1][0]
            _cdiv(K, _Index(32) * 1)
            for pid0 in range(int(nm) * int(nn)):
                pid = _Index(pid0)
                gid, first = pid // ng, (pid // ng) * 8
                gs = min(nm - first, _Index(8))
                pm, pn = first + pid % gs, (pid % ng) // gs
                offm = [pm * bm + _Index(i) for i in range(bm)]
                offn = [pn * bn + _Index(j) for j in range(bn)]
                im = [v % M if second else v for v in offm]
                jn = [v % N if second else v for v in offn]
                ek = [(_Index(0) * 32 + e) if second else _Index(e) for e in range(32)]
                for step in range(int(nk) + 1):
                    sk = (_Index(step) * 32) * 1 if second else _Index(step) * 32
                    for i in range(bm):
                        for e in range(32):
                            ki = ek[e] + sk
                            address = coord(0, im[i], ek[e]) + _Index(step) * ai
                            if step < nk: access(0, address, ki < K if second else offm[i] < M)
                    for e in range(32):
                        ki = ek[e] + sk
                        for j in range(bn):
                            address = coord(1, ek[e] // 8, jn[j]) + _Index(step) * bi
                            if step < nk: access(1, address, ki < K if second else True)
                for step in range(int(nk)):
                    sk = (_Index(step) * 32) * 1 if second else _Index(step) * 32
                    for e in range(32):
                        ki = ek[e] + sk
                        g = ki // group if second else (_Index(0) if group == K else _Index(step) // (group // 32))
                        for j in range(bn):
                            scale = coord(2, g, jn[j]); zero = coord(3, g, jn[j] // 8)
                            (_Index(ek[e]) % 8) * 4; (jn[j] % 8) * 4
                            access(2, scale, ki < K if second else True)
                            access(3, zero, ki < K if second else True)
                for j in range(bn):
                    jn[j] * strides[2][1]; (jn[j] // 8) * strides[3][1]
                for i in range(bm):
                    for j in range(bn):
                        access(4, coord(4, offm[i], offn[j]), offm[i] < M and offn[j] < N)
    else:
        nk, nn = _cdiv(K, 64), _cdiv(N, 32)
        for kp in range(int(nk)):
            for np in range(int(nn)):
                for i in range(64):
                    k = _Index(kp) * 64 + _Index(i)
                    for j in range(32):
                        n = _Index(np) * 32 + _Index(j)
                        active = k < K and n < N
                        access(5, coord(5, k, n), active)
                        access(1, coord(1, k // 8, n), active)
                        access(3, coord(3, k // group, n // 8), active)
                        access(2, coord(2, k // group, n), active)
                        (k % 8) * 4; (n % 8) * 4


def _check(route, a, q, s, z, group, output=None, scratch=None, device=None, dtype=None):
    if type(group) is not int or group <= 0:
        raise ValueError("group_size must be a positive Python integer")
    if not isinstance(q, torch.Tensor) or q.ndim != 2:
        raise ValueError("packed weight must be rank two")
    K, N = q.shape[0] * 8, q.shape[1]
    if device is None:
        if not isinstance(a, torch.Tensor): raise TypeError("input must be a Tensor")
        if a.ndim != 2: raise ValueError("input must be rank two")
        device = a.device
    else:
        device = torch.device(device)
        if device.type != 'cuda': raise ValueError("dequantization requires a CUDA device")
        if device.index is None: device = torch.device('cuda', torch.cuda.current_device())
    selected = torch.float32 if route in ('s2', 'dequant32') else torch.float16
    if dtype is not None and dtype != selected:
        raise ValueError("dequantization dtype must be float16 or float32 and match scales")
    spans = [_dense(q, torch.int32, 'qweight', device), _dense(s, selected, 'scales', device),
             _dense(z, torch.int32, 'qzeros', device)]
    if K <= 0 or N <= 0 or K % 8 or N % 8 or K % group:
        raise ValueError("positive K/N multiples of eight and group_size dividing K are required")
    if tuple(s.shape) != (K // group, N) or tuple(z.shape) != (K // group, N // 8):
        raise ValueError("scale or packed-zero shape mismatch")
    M = 1 if a is None else a.shape[0]
    capacities = [M*K, spans[0][2], spans[1][2], spans[2][2], M*N, K*N]
    if a is not None:
        span = _dense(a, selected if route == 's2' else torch.float16, 'input', device)
        spans.append(span); capacities[0] = span[2]
        if tuple(a.shape) != (M, K): raise ValueError("input shape mismatch")
    if output is not None:
        span = _dense(output, torch.float16, 'output', device);spans.append(span);capacities[4] = span[2]
        if tuple(output.shape) != (M, N): raise ValueError("output shape mismatch")
    if scratch is not None:
        span = _dense(scratch, selected, 'dequantized output', device);spans.append(span);capacities[5] = span[2]
        if tuple(scratch.shape) != (K, N): raise ValueError("dequantized output shape mismatch")
    _apart(spans)
    if route == 'gptq' and (K % 32 or group % 32 or N % 256):
        raise ValueError("both GPTQ tunings require K/group multiples of 32 and N of 256")
    _source_indices(route, M, N, K, group, tuple(capacities))
    return M, N, K


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # Additional configurations are omitted for brevity...
    ],
    key=['M', 'N', 'K', 'NO_GROUPS'],
)
@triton.jit
def matmul4_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales_g, stride_scales_n,
    stride_zeros_g, stride_zeros_n,
    groupsize, NO_GROUPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute the matrix multiplication C = A x B.
    A is of shape (M, K) float16
    B is of shape (K//8, N) int32
    C is of shape (M, N) float16
    scales is of shape (G, N) float16
    zeros is of shape (G, N//8) int32
    groupsize is an int specifying the size of groups for scales and zeros.
    G is K // groupsize.
    Set NO_GROUPS to groupsize == K, in which case G = 1 and the kernel is more efficient.
    WARNING: This kernel assumes that K is a multiple of BLOCK_SIZE_K.
    WARNING: This kernel assumes that N is a multiple of BLOCK_SIZE_N.
    WARNING: This kernel assumes that groupsize is a multiple of BLOCK_SIZE_K.
    """
    bits = 4
    infearure_per_bits = 8
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)   # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    a_mask = (offs_am[:, None] < M)
    b_ptrs = b_ptr + ((offs_k[:, None] // infearure_per_bits) * stride_bk + offs_bn[None, :] * stride_bn)   # (BLOCK_SIZE_K, BLOCK_SIZE_N)
    scales_ptrs = scales_ptr + offs_bn * stride_scales_n   # (BLOCK_SIZE_N,)
    zeros_ptrs = zeros_ptr + ((offs_bn // infearure_per_bits) * stride_zeros_n)   # (BLOCK_SIZE_N,)
    shifter = (offs_k % infearure_per_bits) * bits
    zeros_shifter = (offs_bn % infearure_per_bits) * bits
    if NO_GROUPS:
        scales = tl.load(scales_ptrs)  # (BLOCK_SIZE_N,)
        zeros = tl.load(zeros_ptrs)  # (BLOCK_SIZE_N,), each element is repeated 8 times, int32    
        zeros = (zeros >> zeros_shifter) & 0xF  # (BLOCK_SIZE_N,) int32
        zeros = zeros * scales
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, num_pid_k):
        a = tl.load(a_ptrs, mask=a_mask, other=0.)   # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_ptrs)   # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated    
        if not NO_GROUPS:
            g_id = k // (groupsize // BLOCK_SIZE_K)
            ptr = scales_ptrs + g_id * stride_scales_g
            scales = tl.load(ptr)  # (BLOCK_SIZE_N,)
            ptr = zeros_ptrs + g_id * stride_zeros_g   # (BLOCK_SIZE_N,)
            zeros = tl.load(ptr)  # (BLOCK_SIZE_N,), each element is repeated 8 times, int32    
            zeros = (zeros >> zeros_shifter) & 0xF  # (BLOCK_SIZE_N,) int32
            zeros = (zeros) * scales  # (BLOCK_SIZE_N,) float16    
        b = (b >> shifter[:, None]) & 0xF  # Extract the 4-bit values
        b = b * scales[None, :] - zeros[None, :]  # Scale and shift
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk  
    c = accumulator.to(tl.float16)  
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def matmul_dequantize_int4_gptq(x: torch.FloatTensor, qweight: torch.IntTensor, scales: torch.FloatTensor, qzeros: torch.IntTensor, group_size, output=None) -> torch.FloatTensor:
    """
    Compute the matrix multiplication C = A x B + bias.
    Where B is quantized using GPTQ and groupsize = -1 into 4-bit values.

    A is of shape (..., K) float16
    qweight is of shape (K//8, N) int32
    scales is of shape (G, N) float16
    qzeros is of shape (G, N//8) int32
    bias is of shape (1, N) float16

    groupsize is the number of infeatures in each group.
    G = K // groupsize

    Returns C of shape (..., N) float16
    """
    M, N, K = _check("gptq", x, qweight, scales, qzeros, group_size, output)

    if output is None:
        inplace = False
        output = torch.empty((M, N), device=x.device, dtype=torch.float16)
    else:
        inplace = True

    _check("gptq", x, qweight, scales, qzeros, group_size, output)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    matmul4_kernel[grid](
        x, qweight, output,
        scales, qzeros,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        output.stride(0), output.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        group_size, group_size == K,
    )
    if not inplace:
        return output


@triton.autotune(
    configs=[
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # Additional configurations are omitted for brevity...
    ],
    key=['M', 'N', 'K'],
    reset_to_zero=['c_ptr']
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    bs_ptr, bzp_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bsk, stride_bsn,
    stride_bzpk, stride_bzpn,
    group_size,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr
    ):
    """
    Matrix multiplication kernel with dequantization for 4-bit integers
    A: [M, K], B: [K//8, N], C: [M, N]
    """
    pid = tl.program_id(axis=0)
    pid_sp_k = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + (offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        bs_ptrs = bs_ptr + ((offs_k[:, None] + k * BLOCK_SIZE_K * SPLIT_K) // group_size) * stride_bsk \
            + offs_bn[None, :] * stride_bsn
        bzp_ptrs = bzp_ptr + ((offs_k[:, None] + k * BLOCK_SIZE_K * SPLIT_K) // group_size) * stride_bzpk \
            + (offs_bn[None, :] // 8) * stride_bzpn
        b_shift_bits = (offs_k[:, None] % 8) * 4 
        bzp_shift_bits = (offs_bn[None, :] % 8) * 4
        a = tl.load(a_ptrs, mask=(offs_k[None, :] + k * BLOCK_SIZE_K * SPLIT_K < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k * BLOCK_SIZE_K * SPLIT_K < K), other=0.0)
        bs = tl.load(bs_ptrs, mask=(offs_k[:, None] + k * BLOCK_SIZE_K * SPLIT_K < K), other=0.0)
        bzp = tl.load(bzp_ptrs, mask=(offs_k[:, None] + k * BLOCK_SIZE_K * SPLIT_K < K), other=0.0)

        int_b = (b >> b_shift_bits) & 0xF
        int_bzp = (bzp >> bzp_shift_bits) & 0xF
        b = ((int_b - int_bzp) * bs).to(tl.float16)
        accumulator += tl.dot(a.to(tl.float16), b.to(tl.float16))
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K * SPLIT_K * stride_bk // 8)

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)


def matmul_dequantize_int4_s2(x: torch.FloatTensor, qweight: torch.IntTensor, scales: torch.FloatTensor, qzeros: torch.IntTensor, group_size: int = 128, output=None) -> torch.FloatTensor:
    """
    Perform matrix multiplication and dequantize the result from 4-bit integers.
    """
    M, N, K = _check("s2", x, qweight, scales, qzeros, group_size, output)
    if output is None:
        output = torch.zeros((M, N), device=x.device, dtype=torch.float16)  
    _check("s2", x, qweight, scales, qzeros, group_size, output)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        META['SPLIT_K'],
    )
    matmul_kernel[grid](
        x, qweight, output,
        scales, qzeros,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        output.stride(0), output.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        group_size,
    )
    return output


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        # Additional configurations are omitted for brevity...
    ],
    key=['K', 'N'],
)
@triton.jit
def dequantize_kernel(
    # Pointers to matrices
    b_ptr, b_scale_ptr, b_zp_ptr, fpb_ptr,
    # Matrix dimensions
    K, N, group_size,
    stride_bk, stride_bn,
    stride_bsk, stride_bsn,
    stride_bzpk, stride_bzpn,
    stride_fpbk, stride_fpbn,
    # Meta-parameters
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """Dequantize tile [BLOCK_SIZE_K, BLOCK_SIZE_N] in full precision.
    We should assert BLOCK_SIZE_N % 8 == 0.
    weight[K // 8, N], scale[K // group_size, N], zp[K // group_size, N // group_size]
    """
    k_block_idx = tl.program_id(axis=0)
    n_block_idx = tl.program_id(axis=1)
    offs_k = k_block_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = n_block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    fpb_offs = offs_k[:, None] * stride_fpbk + offs_n[None, :] * stride_fpbn
    b_offs = (offs_k[:, None] // 8) * stride_bk + offs_n[None, :] * stride_bn
    bzp_offs = (offs_k[:, None] // group_size) * stride_bzpk + (offs_n[None, :] // 8) * stride_bzpn
    bs_offs = (offs_k[:, None] // group_size) * stride_bsk + offs_n[None, :] * stride_bsn
    n_mask = offs_n[None, :] < N
    k_mask = offs_k[:, None] < K
    mask = n_mask & k_mask
    int32_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
    zp_b = tl.load(b_zp_ptr + bzp_offs, mask=mask, other=0.0)
    scale_b = tl.load(b_scale_ptr + bs_offs, mask=mask, other=0.0)
    b_shift = (offs_k[:, None] % 8) * 4
    bzp_shift = (offs_n[None, :] % 8) * 4
    fp_weight = (((int32_b >> b_shift) & 0xF) - ((zp_b >> bzp_shift) & 0xF)) * scale_b
    tl.store(fpb_ptr + fpb_offs, fp_weight, mask=mask)


def dequantize_int4(b, b_scale, b_zero_point, device, dtype, group_size):
    route = "dequant32" if dtype == torch.float32 else "dequant16"
    _, N, K = _check(route, None, b, b_scale, b_zero_point, group_size, device=device, dtype=dtype)
    fp_b = torch.ones((K, N), device=device, dtype=dtype)
    _check(route, None, b, b_scale, b_zero_point, group_size, scratch=fp_b, device=device, dtype=dtype)
    grid = lambda META: (
        triton.cdiv(K, META['BLOCK_SIZE_K']),
        triton.cdiv(N, META['BLOCK_SIZE_N']), 
    )
    dequantize_kernel[grid](
        b, b_scale, b_zero_point, fp_b,
        K, N, group_size,
        b.stride(0), b.stride(1),
        b_scale.stride(0), b_scale.stride(1),
        b_zero_point.stride(0), b_zero_point.stride(1),
        fp_b.stride(0), fp_b.stride(1)
    )
    return fp_b


def matmul_dequantize_int4_s1(a, b, b_scale, b_zero_point, group_size=128, out=None):
    """
    Matmul dequantize int4 s1 dequantize weight to `fp_b` and do fp16 torch.mm,
    this is for `prefill` stage, since weight size is fixed so is dequantize overhead,
    perfill stage have more tokens to amortize dequant cost.
    """
    M, N, K = _check("s1", a, b, b_scale, b_zero_point, group_size, out)
    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    fp_b = dequantize_int4(b, b_scale, b_zero_point, a.device, a.dtype, group_size)
    _check("s1", a, b, b_scale, b_zero_point, group_size, out, fp_b)
    torch.mm(a, fp_b, out=out)
    fp_b = None
    return out
