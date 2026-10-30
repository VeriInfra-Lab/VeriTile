
import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time


def get_configs_io_bound():
    def init_to_zero(name):
        return lambda nargs: nargs[name].zero_()

    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k, "SPLIT_K": 1},
                            num_stages=num_stages,
                            num_warps=num_warps,
                        ),
                    )
                    # split_k
                    for split_k in [2, 4, 8, 16]:
                        configs.append(
                            triton.Config(
                                {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k, "SPLIT_K": split_k},
                                num_stages=num_stages,
                                num_warps=num_warps,
                                pre_hook=init_to_zero("C"),
                            ),
                        )
    return configs

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1}, num_stages=5, num_warps=2),
        *get_configs_io_bound(),
    ],
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune, "perf_model": estimate_matmul_time, "top_k": 10},
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    },
)
@triton.jit
def _int8_matmul_rowwise_dequantize(
    A,
    B,
    C,
    bias,
    state_x_ptr,
    state_w_ptr,
    M,
    N,
    K,
    divfactor,
    has_bias: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    pid_z = tl.program_id(1).to(tl.int64)
    M = tl.cast(M, tl.int64)
    N = tl.cast(N, tl.int64)
    K = tl.cast(K, tl.int64)
    stride_am = tl.cast(stride_am, tl.int64)
    stride_ak = tl.cast(stride_ak, tl.int64)
    stride_bk = tl.cast(stride_bk, tl.int64)
    stride_bn = tl.cast(stride_bn, tl.int64)
    stride_cm = tl.cast(stride_cm, tl.int64)
    stride_cn = tl.cast(stride_cn, tl.int64)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    w_factor = tl.load(state_w_ptr + rbn)[None, :]
    x_factor = tl.load(state_x_ptr + ram)[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=0.0)
        acc += tl.dot(a, b)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = w_factor * (x_factor * (acc * divfactor))
    acc = acc.to(C.dtype.element_ty)
    if has_bias:
        if pid_z == 0:
            bias_value = tl.load(bias + rn, mask=rn < N, other=0.0).to(C.dtype.element_ty)
            acc = acc + bias_value[None, :]
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)

def _check(a, b, state_x, state_w, bias):
    tensors = (a, b, state_x, state_w)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("a, b and scales must be tensors")
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise TypeError("bias must be a tensor or None")
    if a.ndim != 2 or b.ndim != 2 or state_x.ndim != 1 or state_w.ndim != 1:
        raise ValueError("expected matrices and scale vectors")
    if any(t.layout != torch.strided for t in tensors):
        raise ValueError("strided tensors required")
    if a.dtype != torch.int8 or b.dtype != torch.int8 or state_x.dtype != torch.float32 or state_w.dtype != torch.float32:
        raise TypeError("matrices must be INT8 and scales FP32")
    if not a.is_cuda or any(t.device != a.device for t in tensors):
        raise ValueError("one CUDA device required")
    if any(not t.is_contiguous() or t.is_neg() for t in tensors):
        raise ValueError("contiguous inputs without negative views required")
    M, K = a.shape
    _, N = b.shape
    if b.shape[0] != K or state_x.shape != (M,) or state_w.shape != (N,):
        raise ValueError("incompatible dimensions")
    if bias is not None:
        if bias.layout != torch.strided or bias.dtype != torch.float32 or bias.shape != (N,) or bias.device != a.device:
            raise ValueError("bias must be an FP32 vector on the input device")
        if not bias.is_contiguous() or bias.is_neg():
            raise ValueError("bias must be contiguous without a negative view")
    floats = (state_x, state_w) + (() if bias is None else (bias,))
    if any(torch._C._overlaps(i, f) for i in (a, b) for f in floats):
        raise ValueError("INT8 and FP32 inputs must not share raw storage")
    if any(t.numel() and t.data_ptr() % 4 for t in floats):
        raise ValueError("FP32 input pointers must be aligned to four bytes")
    if M and N:
        if max(M, N, K) >= 2**61:
            raise ValueError("dimensions exceed native index range")
        if triton.cdiv(M, 16) * triton.cdiv(N, 32) > 2**31 - 1:
            raise ValueError("autotune grid exceeds CUDA grid range")
        # Envelopes include every masked lane and final unused pointer advance.
        bias_offset = 0 if bias is None else bias.storage_offset()
        if (a.storage_offset() + M*K + K + 4095 >= 2**63 or
            b.storage_offset() + (K+4095)*N + N >= 2**63 or
            2*((M+255)*N + N+255) >= 2**63 or
            4*(state_x.storage_offset()+M) >= 2**63 or
            4*(state_w.storage_offset()+N) >= 2**63 or
            4*(bias_offset+N+255) >= 2**63):
            raise ValueError("formed element or byte offsets exceed native range")

def int8_matmul_rowwise_dequantize(a, b, state_x, state_w, bias):
    _check(a, b, state_x, state_w, bias)
    divfactor = 1.0 / (127.0 * 127.0)
    has_bias = 0 if bias is None else 1
    device = a.device
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=device, dtype=torch.float16)
    ACC_TYPE = tl.float32
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), META["SPLIT_K"])
    if M == 0 or N == 0:
        return c
    with torch.cuda.device(device):
        _int8_matmul_rowwise_dequantize[grid](
            a,
            b,
            c,
            bias,
            state_x,
            state_w,
            M,
            N,
            K,
            divfactor,
            has_bias,
            K,
            1,
            N,
            1,
            N,
            1,
            GROUP_M=8,
            ACC_TYPE=ACC_TYPE,
        )
    return c
