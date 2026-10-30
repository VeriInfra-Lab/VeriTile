"""Guarded FP32 argmax: direct global input; original dimension contiguous()."""
import torch
import triton
import triton.language as tl
import math


_MAX_PROGRAMS = 65535
_I64 = 1 << 63
_U64 = 1 << 64
_MAX_BLOCK = 1 << 20
_MID_DTYPES = tuple(getattr(torch, name) for name in (
    "float16", "bfloat16", "float32", "float64", "int8", "int16", "int32",
    "int64", "uint8", "uint16", "uint32", "uint64", "bool"))


def _check_input(inp, dim, keepdim):
    if not isinstance(inp, torch.Tensor):
        raise TypeError("input must be a Tensor")
    if inp.layout != torch.strided or inp.dtype != torch.float32:
        raise TypeError("input must be strided FP32")
    if not inp.is_cuda:
        raise ValueError("input must be CUDA")
    if type(keepdim) is not bool:
        raise TypeError("keepdim must be bool")
    if dim is not None and type(dim) is not int:
        raise TypeError("dim must be an integer or None")
    if dim is None and (not inp.is_contiguous() or inp.is_neg()):
        raise ValueError("global input must be contiguous with no negative view bit")
    if 4 * inp.numel() >= _I64:
        raise ValueError("input extent exceeds signed64 native arithmetic")


def _check_global_geometry(size, block, count, mid_block):
    if not (0 < block <= _MAX_BLOCK and 0 < count <= mid_block <= _MAX_BLOCK):
        raise ValueError("global reduction blocks exceed the native tile domain")
    if size + block >= _I64 or 8 * mid_block >= _I64:
        raise ValueError("global padded indices exceed signed64")


def _check_dimension_geometry(outer, extent, inner):
    if outer == 0 or inner == 0:
        return
    if not (extent > 0 and 4 * outer * extent * inner < _I64
            and 8 * outer * inner < _I64 and outer + 128 < _I64
            and extent + 128 < _I64 and inner < _I64
            and triton.cdiv(outer, 128) * inner < _I64):
        raise ValueError("dimension indices exceed signed64")


def _check_pointer(tensor, alignment):
    if tensor.numel() == 0:
        return
    pointer = tensor.data_ptr()
    if pointer % alignment or pointer + tensor.element_size() * tensor.numel() > _U64:
        raise ValueError("native data pointer is unaligned or its span exceeds uint64")


def _check_fresh(inp, *allocated):
    # Ordinary allocator results, including squeeze views, retain their storage.
    objects = (inp,) + allocated
    spans = []
    for tensor in objects:
        _check_pointer(tensor, min(tensor.element_size(), 8))
        storage = tensor.untyped_storage()
        spans.append((storage.data_ptr(), storage.data_ptr() + storage.nbytes()))
    for i in range(1, len(spans)):
        if any(not (spans[i][1] <= spans[j][0] or spans[j][1] <= spans[i][0])
               for j in range(i)):
            raise ValueError("allocator returned overlapping live storage")


def _launch_segments(programs):
    # Per-launch geometry only: every logical program is retained.
    for first in range(0, programs, _MAX_PROGRAMS):
        yield first, (min(programs - first, _MAX_PROGRAMS),)


def can_use_int32_index(tensor):
    # This function checks if the tensor can use int32 indices
    return tensor.numel() < 2**31

# Kernel 1: argmax_kernel_1_repaired
@triton.jit
def argmax_kernel_1_repaired(
    inp,
    mid_value,
    mid_index,
    M: tl.int64,
    program_base: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    INT64_INDEX: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int64) + program_base
    if INT64_INDEX:
        pid = pid.to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    max_val, max_index = tl.max(inp_val, axis=0, return_indices=True)
    max_index = max_index.to(tl.int64) + pid.to(tl.int64) * BLOCK_SIZE
    mid_value_ptr = mid_value + pid
    max_index_ptr = mid_index + pid
    stored_value = max_val
    if mid_value.dtype.element_ty.is_int():
        is_bool: tl.constexpr = mid_value.dtype.element_ty.is_int1()
        width: tl.constexpr = 8 if is_bool else mid_value.dtype.element_ty.primitive_bitwidth
        signed: tl.constexpr = is_bool or mid_value.dtype.element_ty.is_int_signed()
        exponent: tl.constexpr = width - 1 if signed else width
        lower: tl.constexpr = -(1 << exponent) if signed else 0
        gap: tl.constexpr = (1 << (exponent - 24)) if exponent > 24 else 1
        upper: tl.constexpr = (1 << exponent) - gap
        stored_value = tl.where(max_val == max_val, max_val, 0.0)
        stored_value = tl.minimum(
            tl.maximum(stored_value, tl.full((), lower, tl.float32)),
            tl.full((), upper, tl.float32),
        )
    tl.store(mid_value_ptr, stored_value)
    tl.store(max_index_ptr, max_index)

# Kernel 2: argmax_kernel_2_repaired
@triton.jit
def argmax_kernel_2_repaired(mid_value, mid_index, out, mid_size: tl.int64, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID).to(tl.int64)
    mid_ptrs = mid_value + offset
    mask = offset < mid_size
    if mid_value.dtype.element_ty.is_int1():
        padding: tl.constexpr = 0
    elif mid_value.dtype.element_ty.is_int():
        width: tl.constexpr = mid_value.dtype.element_ty.primitive_bitwidth
        if mid_value.dtype.element_ty.is_int_signed():
            padding: tl.constexpr = -(1 << (width - 1))
        else:
            padding: tl.constexpr = 0
    else:
        padding: tl.constexpr = -float("inf")
    mid_val = tl.load(mid_ptrs, mask=mask, other=padding)
    index_val = tl.argmax(mid_val, axis=0).to(tl.int64)
    mid_index_ptrs = mid_index + index_val
    out_val = tl.load(mid_index_ptrs)
    tl.store(out, out_val)

# Kernel 3: argmax_kernel
@triton.jit
def argmax_kernel_repaired(
    inp,
    out_index,
    M: tl.int64,
    N: tl.int64,
    K: tl.int64,
    program_base: tl.int64,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INT64_INDEX: tl.constexpr = False,
):
    program = tl.program_id(0).to(tl.int64) + program_base
    pid_m = program // K
    pid_k = program % K
    if INT64_INDEX:
        pid_m = pid_m.to(tl.int64)
        pid_k = pid_k.to(tl.int64)
    m_offset = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)

    max_values = tl.full([BLOCK_M], dtype=tl.float32, value=float("-inf"))
    argmax_values = tl.full([BLOCK_M], dtype=tl.int64, value=0)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N).to(tl.int64)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k.to(tl.int64)
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
        local_max, local_argmax = tl.max(
            inp_vals, 1, return_indices=True, return_indices_tie_break_left=True
        )
        update = local_max > max_values
        max_values = tl.where(update, local_max, max_values)
        argmax_values = tl.where(update, start_n + local_argmax.to(tl.int64), argmax_values)

    offset_index = m_offset * K + pid_k.to(tl.int64)
    out_index_ptrs = out_index + offset_index
    mask1 = m_offset < M
    tl.store(out_index_ptrs, argmax_values, mask=mask1)

# Function calling the kernels
def argmax(inp, dim=None, keepdim=False, *, dtype=None):
    _check_input(inp, dim, keepdim)
    if dim is None:
        M = inp.numel()
        if M == 0:
            raise IndexError("argmax requires a nonempty reduction")
        if dtype is None:
            dtype = inp.dtype
        if dtype not in _MID_DTYPES:
            raise TypeError("unsupported global intermediate dtype")
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        _check_global_geometry(M, block_size, mid_size, block_mid)
        _check_pointer(inp, 4)
        use_int64_index = not can_use_int32_index(inp)

        mid_value = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        mid_index = torch.zeros((block_mid,), dtype=torch.int64, device=inp.device)
        if keepdim:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=torch.int64, device=inp.device)
        else:
            out = torch.empty([], dtype=torch.int64, device=inp.device)

        _check_fresh(inp, mid_value, mid_index, out)
        with torch.cuda.device(inp.device):
            for first, grid in _launch_segments(mid_size):
                argmax_kernel_1_repaired[grid](
                    inp,
                    mid_value,
                    mid_index,
                    M,
                    first,
                    block_size,
                    INT64_INDEX=use_int64_index,
                    num_ctas=1,
                )
            argmax_kernel_2_repaired[(1, 1, 1)](
                mid_value, mid_index, out, mid_size, block_mid, num_ctas=1,
            )
        return out
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        shape = inp.shape
        dim = dim % inp.ndim
        N = shape[dim]
        M = math.prod(shape[:dim])
        K = math.prod(shape[dim + 1:])
        if N == 0:
            raise IndexError("argmax requires a nonempty reduction axis")

        _check_dimension_geometry(M, N, K)
        inp = inp.contiguous()
        use_int64_index = not can_use_int32_index(inp)

        shape_list = list(shape)
        shape_list[dim] = 1
        out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)
        if not keepdim:
            out_index = torch.squeeze(out_index, dim)

        if M == 0 or K == 0:
            return out_index

        if inp.is_neg():
            raise ValueError("native input must have no negative view bit")
        _check_pointer(inp, 4)
        _check_fresh(inp, out_index)

        BLOCK_M = 128  # Exact source tile
        BLOCK_N = 128  # Example, adjust as needed

        programs = triton.cdiv(M, BLOCK_M) * K
        with torch.cuda.device(inp.device):
            for first, grid in _launch_segments(programs):
                argmax_kernel_repaired[grid](
                    inp,
                    out_index,
                    M,
                    N,
                    K,
                    first,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    INT64_INDEX=use_int64_index,
                    num_ctas=1,
                )

        return out_index


argmax_repaired = argmax
