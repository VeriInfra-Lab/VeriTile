from typing import Optional, Union
import torch
import triton
import triton.language as tl

@triton.jit
def rotary_kernel(
    OUT, X, COS, SIN, CU_SEQLENS, SEQLEN_OFFSETS, seqlen, nheads, rotary_dim, seqlen_ro,
    CACHE_KEY_SEQLEN, stride_out_batch, stride_out_seqlen, stride_out_nheads,
    stride_out_headdim, stride_x_batch, stride_x_seqlen, stride_x_nheads,
    stride_x_headdim, BLOCK_K: tl.constexpr, IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr, INTERLEAVED: tl.constexpr, CONJUGATE: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)
    rotary_dim_half = rotary_dim // 2

    if not IS_VARLEN:
        X = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
        OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        X = X + start_idx * stride_x_seqlen + pid_head * stride_x_nheads
        OUT = OUT + start_idx * stride_out_seqlen + pid_head * stride_out_nheads

    if pid_m * BLOCK_M >= seqlen:
        return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)
    rk = tl.arange(0, BLOCK_K)
    rk_half = tl.arange(0, BLOCK_K // 2)

    if not INTERLEAVED:
        X = X + (rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        cos = tl.load(COS, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=1.0).to(tl.float32)
        sin = tl.load(SIN, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=0.0).to(tl.float32)
        x0 = tl.load(X, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0).to(tl.float32)
        x1 = tl.load(X + rotary_dim_half * stride_x_headdim, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim)
        tl.store(OUT, o0, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
        tl.store(OUT + rotary_dim_half * stride_out_headdim, o1, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
    else:
        rk_swap = rk + ((rk + 1) % 2) * 2 - 1
        rk_repeat = tl.arange(0, BLOCK_K) // 2
        X0 = X + (rm[:, None] * stride_x_seqlen + rk[None, :] * stride_x_headdim)
        X1 = X + (rm[:, None] * stride_x_seqlen + rk_swap[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        cos = tl.load(COS, mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half), other=1.0).to(tl.float32)
        sin = tl.load(SIN, mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half), other=0.0).to(tl.float32)
        x0 = tl.load(X0, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim), other=0.0).to(tl.float32)
        x1 = tl.load(X1, mask=(rm[:, None] < seqlen) & (rk_swap[None, :] < rotary_dim), other=0.0).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        x0_cos = x0 * cos
        x1_sin = x1 * sin
        out = tl.where(rk[None, :] % 2 == 0, x0_cos - x1_sin, x0_cos + x1_sin)
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk[None, :] * stride_out_headdim)
        tl.store(OUT, out, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim))

def _storage_span(value):
    if value.numel() == 0:
        return None
    storage = value.untyped_storage()
    start, size, unit = storage.data_ptr(), storage.nbytes(), value.element_size()
    last = sum((n-1)*s for n,s in zip(value.shape,value.stride()))
    if (value.data_ptr() != start or start % unit or unit*(last+1) > size
            or start+size > 2**64):
        raise ValueError("aligned contained payload and metadata storage is required")
    return start,start+size


def _validate_inputs(x, cos, sin, offsets, cumulative, maximum, interleaved, inplace, conjugate):
    if any(type(flag) is not bool for flag in (interleaved, inplace, conjugate)):
        raise ValueError("rotation flags must be bool")
    if not isinstance(x, torch.Tensor):
        raise ValueError("x must be a Tensor")
    device = x.device
    tensors = [x, cos, sin]
    for value in tensors:
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided or not value.is_cuda or value.device != device
                or value.dtype != torch.float32 or value.storage_offset() != 0
                or value.is_neg() or value.is_conj() or any(s <= 0 for s in value.stride())):
            raise ValueError("expected ordinary FP32 CUDA tensors on one device")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if x.numel():
        expected_stride = 1
        for size, actual_stride in zip(reversed(x.shape), reversed(x.stride())):
            if actual_stride != expected_stride:
                raise ValueError("nonempty x must have canonical contiguous strides")
            expected_stride *= size
    if x.ndim != (3 if cumulative is not None else 4) or cos.ndim != 2 or sin.shape != cos.shape:
        raise ValueError("incompatible input or coefficient shapes")
    heads, width = x.shape[-2:]
    rotary = 2 * cos.shape[1]
    if heads <= 0 or width <= 0 or rotary > width:
        raise ValueError("positive heads/width and rotary width <= head width are required")
    if cumulative is None:
        batch, longest = x.shape[:2]
    else:
        if (not isinstance(cumulative, torch.Tensor) or cumulative.layout != torch.strided or not cumulative.is_cuda
                or cumulative.device != device or cumulative.ndim != 1 or cumulative.numel() < 1
                or cumulative.dtype not in (torch.int32, torch.int64)
                or cumulative.storage_offset() != 0 or cumulative.stride(0) != 1
                or cumulative.is_neg() or cumulative.is_conj()):
            raise ValueError("CU must be a unit-stride int32/int64 CUDA vector")
        if type(maximum) is not int or maximum < 0:
            raise ValueError("packed inputs require a nonnegative integer max_seqlen")
        _storage_span(cumulative)
        cu = cumulative.to(device="cpu").tolist()
        batch, longest = len(cu) - 1, maximum
        if cu[0] != 0 or cu[-1] != x.shape[0] or any(b < a or b-a > longest for a,b in zip(cu,cu[1:])):
            raise ValueError("CU must partition the packed input into sequences bounded by max_seqlen")
        tensors.append(cumulative)
    if isinstance(offsets, torch.Tensor):
        if (offsets.layout != torch.strided or not offsets.is_cuda or offsets.device != device or offsets.shape != (batch,)
                or offsets.dtype not in (torch.int32, torch.int64) or offsets.storage_offset() != 0
                or offsets.stride(0) <= 0 or offsets.is_neg() or offsets.is_conj()):
            raise ValueError("offsets must be an ordinary positive-stride int32/int64 CUDA vector")
        _storage_span(offsets)
        shifts = offsets.to(device="cpu").tolist()
        tensors.append(offsets)
    else:
        if type(offsets) is not int:
            raise ValueError("scalar sequence offset must be an integer")
        shifts = [offsets]
    if any(v < 0 for v in shifts):
        raise ValueError("sequence offsets must be nonnegative")
    block = triton.next_power_of_2(max(32, rotary))
    largest = max(shifts, default=0)
    if (any(v.numel() >= 2**31 for v in tensors) or batch+1 >= 2**31
            or longest+8 >= 2**31 or largest+longest+8 >= 2**31
            or (largest+longest+8)*cos.shape[1]+block >= 2**31
            or x.numel()+8*heads*width+block >= 2**31):
        raise ValueError("source address arithmetic exceeds the admitted signed32 domain")
    spans = [span for value in tensors if (span := _storage_span(value)) is not None]
    if any(a < d and c < b for i,(a,b) in enumerate(spans) for c,d in spans[i+1:]):
        raise ValueError("input backing allocations must be physically separate")


def _check_launch_pointers(output, x, cos, sin, cu, offsets, longest, heads, width, block):
    """Includes every formed padded address, not only active loads/stores."""
    shifts = offsets.to(device="cpu").tolist() if isinstance(offsets, torch.Tensor) else [offsets]
    bound_data = x.numel() + 8*heads*width + block
    bound_coeff = (max(shifts,default=0)+longest+8)*cos.shape[1] + block
    for tensor,bound in ((output,bound_data),(x,bound_data),(cos,bound_coeff),(sin,bound_coeff)):
        # Even an empty coefficient cache forms masked pointers in the kernel.
        if tensor.data_ptr()+4*bound > 2**64:
            raise ValueError("formed source pointers must fit unsigned int64")
    if output.numel() and output.data_ptr()%4:
        raise ValueError("output pointer must be four-byte aligned")

def apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None, max_seqlen: Optional[int] = None,
    interleaved=False, inplace=False, conjugate=False
) -> torch.Tensor:
    _validate_inputs(x, cos, sin, seqlen_offsets, cu_seqlens, max_seqlen, interleaved, inplace, conjugate)
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        total_seqlen, nheads, headdim = x.shape
        batch = cu_seqlens.shape[0] - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim = cos.shape
    rotary_dim *= 2

    cos, sin = cos.contiguous(), sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        seqlen_offsets = seqlen_offsets.contiguous()

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    if x.numel() == 0 or rotary_dim == 0:
        return output

    BLOCK_K = triton.next_power_of_2(max(32, rotary_dim))
    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), batch, nheads)
    BLOCK_M = 4 if interleaved else (8 if rotary_dim <= 64 else 4)
    if BLOCK_M * (BLOCK_K if interleaved else BLOCK_K // 2) > (1 << 20):
        raise ValueError("source tile exceeds Triton static tensor element limit")

    _check_launch_pointers(output, x, cos, sin, cu_seqlens, seqlen_offsets, seqlen, nheads, headdim, BLOCK_K)

    with torch.cuda.device(x.device.index):
        rotary_kernel[grid](
            output, x, cos, sin, cu_seqlens, seqlen_offsets, seqlen, nheads, rotary_dim,
            seqlen_ro, seqlen // 128, output.stride(0) if not is_varlen else 0, output.stride(-3),
            output.stride(-2), output.stride(-1), x.stride(0) if not is_varlen else 0,
            x.stride(-3), x.stride(-2), x.stride(-1), BLOCK_K,
            isinstance(seqlen_offsets, torch.Tensor), is_varlen, interleaved, conjugate, BLOCK_M
        )
    return output
