import time
import torch
import triton
import triton.language as tl


_I32 = 1 << 31
_U64 = 1 << 64


def _tensor_view(t, dtype, name, device=None):
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if t.dtype != dtype or t.layout != torch.strided or t.ndim != 2:
        raise ValueError(f"{name} requires a rank-two strided {dtype} Tensor")
    if t.is_neg() or t.is_conj():
        raise ValueError(f"{name} must have resolved view flags")
    if device is not None and (not t.is_cuda or t.device != device):
        raise ValueError(f"{name} must be on the input CUDA device")
    rows, cols = t.shape
    sr, sc = t.stride()
    if rows <= 0 or cols <= 0 or sr <= 0 or sc <= 0:
        raise ValueError(f"{name} requires positive shape and strides")
    if not (cols * sc <= sr or rows * sr <= sc):
        raise ValueError(f"{name} requires separated strides")
    width = t.element_size()
    storage = t.untyped_storage()
    base, size = storage.data_ptr(), storage.nbytes()
    offset, ptr = t.storage_offset(), t.data_ptr()
    if offset < 0 or ptr != base + width * offset or ptr % width:
        raise ValueError(f"{name} has an invalid or unaligned view origin")
    if base < 0 or base + size >= _U64 or size % width:
        raise ValueError(f"{name} has an invalid allocation span")
    if width * (offset + (rows - 1) * sr + (cols - 1) * sc + 1) > size:
        raise ValueError(f"{name} exceeds its allocation")
    return base, base + size


def _disjoint(spans):
    for i, (lo, hi) in enumerate(spans):
        if any(not (hi <= other_lo or other_hi <= lo)
               for other_lo, other_hi in spans[:i]):
            raise ValueError("all matrix allocations must be disjoint")


def _check_matmul(x, qweight, scales, qzeros, group_size, output=None):
    if type(group_size) is not int or group_size <= 0:
        raise ValueError("group_size must be a positive Python integer")
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        raise ValueError("x must be a CUDA Tensor")
    inputs = (x, qweight, scales, qzeros)
    spans = [_tensor_view(t, dtype, name, x.device) for t, dtype, name in zip(
        inputs, (torch.float16, torch.int32, torch.float16, torch.int32),
        ("x", "qweight", "scales", "qzeros"))]
    M, K = x.shape
    N = qweight.shape[1]
    if K % 128 or N % 256 or group_size % 128 or K % group_size:
        raise ValueError("all 17 tunings require K/group multiples of 128 and N of 256")
    if (tuple(qweight.shape) != (K // 8, N) or
            tuple(scales.shape) != (K // group_size, N) or
            tuple(qzeros.shape) != (K // group_size, N // 8)):
        raise ValueError("packed weight, scale, or zero shape mismatch")
    if not x.is_contiguous() or tuple(x.stride()) != (K, 1):
        raise ValueError("x requires canonical contiguous strides")
    if output is not None:
        spans.append(_tensor_view(output, torch.float16, "output", x.device))
        if tuple(output.shape) != (M, N):
            raise ValueError("output shape mismatch")
    _disjoint(spans)
    views = inputs + (() if output is None else (output,))
    if not (M + 128 < _I32 and 8 * N + 256 < _I32 and
            K + 128 < _I32 and group_size < _I32):
        raise ValueError("dimension exceeds the signed int32 source envelope")
    if any(v.stride(d) >= _I32 for v in views for d in (0, 1)):
        raise ValueError("stride exceeds signed int32")
    a, q, s, z = [v.stride() for v in inputs]
    c = (N, 1) if output is None else output.stride()
    formed = ((M + 128) * a[0] + (K + 128) * a[1],
              (K // 8 + 16) * q[0] + N * q[1],
              (K // group_size) * s[0] + N * s[1],
              (K // group_size) * z[0] + (N // 8) * z[1],
              (M + 128) * c[0] + N * c[1])
    if any(v >= _I32 for v in formed):
        raise ValueError("formed source address exceeds signed int32")
    for t, span in zip(views, spans):
        if t.data_ptr() + t.element_size() * _I32 >= _U64:
            raise ValueError("formed pointer exceeds uint64")
    return M, N, K


def _check_quantize(weight, group_size, tp_rank):
    if type(group_size) is not int or group_size <= 0:
        raise ValueError("group_size must be a positive Python integer")
    if type(tp_rank) is not int or tp_rank < 0:
        raise ValueError("tp_rank must be a nonnegative Python integer")
    _tensor_view(weight, torch.float32, "weight")
    K, N = weight.shape
    if K % 8 or N % 8 or K % group_size:
        raise ValueError("K and N require multiples of 8 and group_size must divide K")
    if weight.untyped_storage().nbytes() >= 1 << 63 or 4 * K * N >= 1 << 63:
        raise ValueError("quantizer address exceeds signed int64")
    _finite(weight, "weight")


def _finite(value, name):
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must be finite")


@triton.autotune(
	configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8), 
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
    # b_ptrs is set up such that it repeats elements along the K axis 8 times
    b_ptrs = b_ptr + ((offs_k[:, None] // infearure_per_bits) * stride_bk + offs_bn[None, :] * stride_bn)   # (BLOCK_SIZE_K, BLOCK_SIZE_N)
    scales_ptrs = scales_ptr + offs_bn * stride_scales_n   # (BLOCK_SIZE_N,)
    # zeros_ptrs is set up such that it repeats elements along the N axis 8 times
    zeros_ptrs = zeros_ptr + ((offs_bn // infearure_per_bits) * stride_zeros_n)   # (BLOCK_SIZE_N,)
    # shifter is used to extract the 4 bits of each element in the 32-bit word from B and zeros
    shifter = (offs_k % infearure_per_bits) * bits
    zeros_shifter = (offs_bn % infearure_per_bits) * bits
    # If G == 1, scales and zeros are the same for all K, so we can load them once
    if NO_GROUPS:
        # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
        scales = tl.load(scales_ptrs)  # (BLOCK_SIZE_N,)
        zeros = tl.load(zeros_ptrs)  # (BLOCK_SIZE_N,), each element is repeated 8 times, int32	
        # Unpack zeros
        zeros = (zeros >> zeros_shifter) & 0xF  # (BLOCK_SIZE_N,) int32
        # zeros = (zeros + 1) * scales  # (BLOCK_SIZE_N,) float16
        zeros = zeros * scales
    # Now calculate a block of output of shape (BLOCK_SIZE_M, BLOCK_SIZE_N)
    # M is along the batch dimension, N is along the outfeatures dimension, K is along the infeatures dimension
    # So this loop is along the infeatures dimension (K)
    # It's calculating BLOCK_SIZE_M batches in parallel, and for each batch, BLOCK_SIZE_N outfeatures in parallel
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, num_pid_k):
        a = tl.load(a_ptrs, mask=a_mask, other=0.)   # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated
        if not NO_GROUPS:
            g_id = k // (groupsize // BLOCK_SIZE_K)
            ptr = scales_ptrs + g_id * stride_scales_g
            scales = tl.load(ptr)  # (BLOCK_SIZE_N,)
            ptr = zeros_ptrs + g_id * stride_zeros_g   # (BLOCK_SIZE_N,)
            zeros = tl.load(ptr)  # (BLOCK_SIZE_N,), each element is repeated 8 times, int32	
            # Unpack zeros
            zeros = (zeros >> zeros_shifter) & 0xF  # (BLOCK_SIZE_N,) int32
            zeros = (zeros) * scales  # (BLOCK_SIZE_N,) float16	
        # Now we need to unpack b (which is 4-bit values) into 32-bit values
        b = (b >> shifter[:, None]) & 0xF  # Extract the 4-bit values
        b = b * scales[None, :] - zeros[None, :]  # Scale and shift
        # print("data type", a, b)
        accumulator += tl.dot(a, b.to(a.dtype))
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk  
    c = accumulator.to(c_ptr.dtype.element_ty)  
    # Store the result
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
	M, N, K = _check_matmul(x, qweight, scales, qzeros, group_size, output)

	# output = torch.empty((M, N), device='cuda', dtype=torch.float16)
	if output is None:
		inplace = False
		output = torch.empty((M, N), device=x.device, dtype=x.dtype)
	else:
		inplace = True

	_check_matmul(x, qweight, scales, qzeros, group_size, output)
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
	# return output
	if not inplace:
		return output

def quantize_int4(weight, group_size=128, tp_rank=0):
    # Weight shape: [H1 // 8, H2]
    # Scale shape: [H1 // group_size, H2]
    # zero_pint shape: [H1 // group_size, H2 // 8]

    _check_quantize(weight, group_size, tp_rank)
    weight = weight.transpose(1, 0)
    h1, h2 = weight.shape
    assert h1 % 8 == 0 and h2 % 8 == 0, "H1 {} H2 {}".format(h1, h2)
    assert h2 % group_size == 0, "H1 {} H2 {}".format(h1, h2)
    weight = weight.contiguous().view(-1, group_size).cuda(tp_rank)
    weight_max = weight.amax(-1, keepdim=True)
    weight_max = torch.where(weight_max < 0, 0, weight_max)
    weight_min = weight.amin(-1, keepdim=True)
    weight_min = torch.where(weight_min > 0, 0, weight_min)
    weight_range = weight_max - weight_min 
    scale = weight_range / (2 ** 4 - 1)
    _finite(scale, "scale")
    if not bool((scale > 0).all().item()):
        raise ValueError("each group requires a strictly positive scale")
    zero_quotient = -weight_min / scale
    _finite(zero_quotient, "zero quotient")
    zero_point = zero_quotient.round().clamp(0, 15).to(torch.int32)
    weight_quotient = weight / scale
    _finite(weight_quotient, "weight quotient")
    weight_shifted = weight_quotient + zero_point
    _finite(weight_shifted, "weight quotient plus zero point")
    weight = weight_shifted.round().clamp(0, 15).to(torch.int32).view(h1, h2)
    int_weight = torch.zeros((h1, h2 // 8), dtype=torch.int32, device=weight.device)
    int_zero_point = torch.zeros(h1 // 8, h2 // group_size).to(torch.int32).to(weight.device)
    zero_point = zero_point.view(h1, -1)
    scale = scale.view(h1, -1)
    # pack 8 int4 in an int32 number.
    # Weight pack in row.
    for pack in range(0, h2, 8):
        for i in range(8):
            int_weight[:, pack // 8] += weight[:, pack + i] << (i * 4)
    # zero point pack in col.
    for pack in range(0, h1, 8):
        for i in range(8):
            int_zero_point[pack // 8, :] += zero_point[pack + i, :] << (i * 4)
    '''
    fp_weight = torch.zeros(h1, h2).half().to(weight.device)
    for pack in range(0, h1 // 8):
        for i in range(8):
            fp_weight[pack * 8 + i, :] = \
                ((int_weight[pack, :] << (28 - i * 4) >> 28) + 16) % 16
    print((fp_weight - weight).abs().sum())

    fp_zp = torch.zeros(zero_point.shape).half().to(zero_point.device)
    for pack in range(0, h1 // 8):
        for i in range(8):
            fp_zp[pack * 8 + i, :] = \
                (int_zero_point[pack, :] >> (i * 4)) & 15

    print((fp_zp - zero_point).abs().sum())
    '''
    weight = None
    return int_weight.transpose(1, 0).contiguous(), scale.transpose(1, 0).contiguous(), int_zero_point.transpose(1, 0).contiguous(), group_size




