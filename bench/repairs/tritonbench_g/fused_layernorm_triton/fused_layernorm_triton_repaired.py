
import torch
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor

@triton.autotune(
    configs=[
        triton.Config(
            {
                "XBLOCK": 1,
                "RBLOCK": 1024,
            },
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {
                "XBLOCK": 1,
                "RBLOCK": 2048,
            },
            num_stages=1,
            num_warps=8,
        ),
    ],
    key=["xnumel", "rnumel"],
)
@triton.jit
def triton_red_fused_native_layer_norm_0(
    in_out_ptr0,
    in_ptr0,
    in_ptr1,
    in_ptr2,
    out_ptr0,
    out_ptr1,
    xnumel,
    rnumel,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    xnumel = tl.cast(xnumel, tl.int64)
    rnumel = tl.cast(rnumel, tl.int64)
    zero = tl.full((), 0, tl.int64)
    step = tl.full((), RBLOCK, tl.int64)
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK).to(tl.int64)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK).to(tl.int64)[None, :]
    x0 = xindex
    tmp3_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(zero, rnumel, step):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(
            in_ptr0 + (r1 + (rnumel * x0)), rmask, eviction_policy="evict_last"
        ).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(rmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(rmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(rmask, tmp3_weight_next, tmp3_weight)
    tmp3_tmp, tmp4_tmp, tmp5_tmp = triton_helpers.welford(
        tmp3_mean, tmp3_m2, tmp3_weight, 1
    )
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp3, None)
    tmp6 = rnumel
    tmp7 = tmp4 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, None)
    for roffset in range(zero, rnumel, step):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(
            in_ptr0 + (r1 + (rnumel * x0)), rmask, eviction_policy="evict_first"
        ).to(tl.float32)
        tmp15 = tl.load(in_ptr1 + (r1), rmask, eviction_policy="evict_last").to(
            tl.float32
        )
        tmp18 = tl.load(in_ptr2 + (r1), rmask, eviction_policy="evict_last").to(
            tl.float32
        )
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp12 - tmp3
        tmp14 = tmp13 * tmp10
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp14 * tmp16
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp17 + tmp19
        tmp21 = tmp20.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (rnumel * x0)), tmp21, rmask)

def _checked_inputs(weight, bias, x):
    tensors = (weight, bias, x)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("weight, bias and input must be Tensors")
    for t in tensors:
        if t.layout != torch.strided or t.dtype != torch.float32:
            raise TypeError("only strided FP32 tensors are supported")
        if not t.is_cuda or t.device.index != 0:
            raise ValueError("inputs must be on CUDA device 0")
        if not t.is_contiguous() or t.storage_offset() != 0:
            raise ValueError("inputs must be contiguous with zero storage offset")
        if t.is_neg():
            raise ValueError("unresolved negative views are unsupported")
        if t.numel() and t.data_ptr() % 4:
            raise ValueError("nonempty FP32 inputs must have 4-byte aligned pointers")
    if x.ndim != 2 or x.shape[1] <= 0:
        raise ValueError("input must have shape [S, D] with D > 0")
    if weight.shape != (x.shape[1],) or bias.shape != (x.shape[1],):
        raise ValueError("weight and bias must each have shape [D]")
    return x.shape


def fused_native_layer_norm(primals_1, primals_2, primals_3):
    S, D = _checked_inputs(primals_1, primals_2, primals_3)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((S, 1), (1, 1), torch.float32)
        buf1 = empty_strided_cuda((S, 1), (1, S), torch.float32)
        buf3 = reinterpret_tensor(buf1, (S, 1), (1, 1), 0)
        del buf1  # reuse
        buf4 = empty_strided_cuda((S, D), (D, 1), torch.bfloat16)
        stream0 = get_raw_stream(0)
        grid = lambda META: (triton.cdiv(S, META["XBLOCK"]),)
        triton_red_fused_native_layer_norm_0[grid](
            buf3, primals_3, primals_1, primals_2, buf0, buf4, S, D
        )
    return (
        buf4,
        primals_3,
        buf0,
        buf3,
    )


# Related candidate: scalarize the constant row values once before normalization.
# No performance measurements or speedup claim are attached to this candidate.
@triton.autotune(
    configs=[
        triton.Config(
            {
                "XBLOCK": 1,
                "RBLOCK": 1024,
            },
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {
                "XBLOCK": 1,
                "RBLOCK": 2048,
            },
            num_stages=1,
            num_warps=8,
        ),
    ],
    key=["xnumel", "rnumel"],
)
@triton.jit
def triton_red_fused_native_layer_norm_hoisted(
    in_out_ptr0,
    in_ptr0,
    in_ptr1,
    in_ptr2,
    out_ptr0,
    out_ptr1,
    xnumel,
    rnumel,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    xnumel = tl.cast(xnumel, tl.int64)
    rnumel = tl.cast(rnumel, tl.int64)
    zero = tl.full((), 0, tl.int64)
    step = tl.full((), RBLOCK, tl.int64)
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK).to(tl.int64)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK).to(tl.int64)[None, :]
    x0 = xindex
    tmp3_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(zero, rnumel, step):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(
            in_ptr0 + (r1 + (rnumel * x0)), rmask, eviction_policy="evict_last"
        ).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(rmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(rmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(rmask, tmp3_weight_next, tmp3_weight)
    tmp3_tmp, tmp4_tmp, tmp5_tmp = triton_helpers.welford(
        tmp3_mean, tmp3_m2, tmp3_weight, 1
    )
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp3, None)
    tmp6 = rnumel
    tmp7 = tmp4 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, None)
    # The single row axis is constant through normalization. Project once,
    # then use one-dimensional column tiles in every normalization chunk.
    xrow = tl.reshape(x0, ())
    row_mean = tl.reshape(tmp3, ())
    row_inverse = tl.reshape(tmp10, ())
    for roffset in range(zero, rnumel, step):
        rindex = roffset + tl.arange(0, RBLOCK).to(tl.int64)
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(
            in_ptr0 + (r1 + (rnumel * xrow)), rmask, eviction_policy="evict_first"
        ).to(tl.float32)
        tmp15 = tl.load(in_ptr1 + (r1), rmask, eviction_policy="evict_last").to(
            tl.float32
        )
        tmp18 = tl.load(in_ptr2 + (r1), rmask, eviction_policy="evict_last").to(
            tl.float32
        )
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp12 - row_mean
        tmp14 = tmp13 * row_inverse
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp14 * tmp16
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp17 + tmp19
        tmp21 = tmp20.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (rnumel * xrow)), tmp21, rmask)


def fused_native_layer_norm_hoisted(primals_1, primals_2, primals_3):
    S, D = _checked_inputs(primals_1, primals_2, primals_3)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((S, 1), (1, 1), torch.float32)
        buf1 = empty_strided_cuda((S, 1), (1, S), torch.float32)
        buf3 = reinterpret_tensor(buf1, (S, 1), (1, 1), 0)
        del buf1  # reuse
        buf4 = empty_strided_cuda((S, D), (D, 1), torch.bfloat16)
        stream0 = get_raw_stream(0)
        grid = lambda META: (triton.cdiv(S, META["XBLOCK"]),)
        triton_red_fused_native_layer_norm_hoisted[grid](
            buf3, primals_3, primals_1, primals_2, buf0, buf4, S, D
        )
    return (
        buf4,
        primals_3,
        buf0,
        buf3,
    )
