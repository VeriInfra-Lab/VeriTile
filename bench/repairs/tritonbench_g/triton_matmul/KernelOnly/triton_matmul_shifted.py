
import torch
import triton
import triton.language as tl


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2. * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_shifted(a_ptr, b_ptr, c_ptr,  #
                  M, N, K,  #
                  stride_am, stride_ak,  #
                  stride_bk, stride_bn,  #
                  stride_cm, stride_cn,  #
                  BLOCK_SIZE_M: tl.constexpr,  #
                  BLOCK_SIZE_N: tl.constexpr,  #
                  BLOCK_SIZE_K: tl.constexpr,  #
                  GROUP_SIZE_M: tl.constexpr,  #
                  ):
    # Widen before all dimension, offset, and stride arithmetic.
    pid = tl.program_id(axis=0).to(tl.int64)
    M, N, K = tl.cast(M, tl.int64), tl.cast(N, tl.int64), tl.cast(K, tl.int64)
    stride_am, stride_ak = tl.cast(stride_am, tl.int64), tl.cast(stride_ak, tl.int64)
    stride_bk, stride_bn = tl.cast(stride_bk, tl.int64), tl.cast(stride_bn, tl.int64)
    stride_cm, stride_cn = tl.cast(stride_cm, tl.int64), tl.cast(stride_cn, tl.int64)
    # Quotient/remainder ceiling avoids overflowing dimension + block - 1.
    num_pid_m = M // BLOCK_SIZE_M + (M % BLOCK_SIZE_M != 0).to(tl.int64)
    num_pid_n = N // BLOCK_SIZE_N + (N % BLOCK_SIZE_N != 0).to(tl.int64)
    if GROUP_SIZE_M == 8:
        num_pid_in_group = num_pid_n << 3
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    lanes_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    lanes_n = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    mask_m = lanes_m < M - start_m
    mask_n = lanes_n < N - start_n
    # Clamp both addends before adding, including inactive output lanes.
    offs_am = tl.where(mask_m, start_m, 0) + tl.where(mask_m, lanes_m, 0)
    offs_bn = tl.where(mask_n, start_n, 0) + tl.where(mask_n, lanes_n, 0)

    # Clamped tail coordinates do not promise full-tile contiguity.
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    blocks_k = K // BLOCK_SIZE_K + (K % BLOCK_SIZE_K != 0).to(tl.int64)
    for k in range(0, blocks_k):
        base_k = k * BLOCK_SIZE_K
        mask_k = offs_k < K - base_k
        safe_k = tl.where(mask_k, base_k, 0) + tl.where(mask_k, offs_k, 0)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + safe_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (safe_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
        accumulator = tl.dot(a, b, accumulator)

    if (c_ptr.dtype.element_ty == tl.float8e4nv):
        c = accumulator.to(tl.float8e4nv)
    else:
        c = accumulator.to(tl.float16)

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_shifted(a, b):
    configs = {
        torch.float8_e4m3fn: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_stages": 4,
            "num_warps": 8
        }, torch.float16: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_stages": 3,
            "num_warps": 8
        }
    }
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("expected Tensor arguments")
    if a.layout != torch.strided or b.layout != torch.strided:
        raise ValueError("expected strided tensors")
    if a.dtype not in (torch.float16, torch.float8_e4m3fn) or a.dtype != b.dtype:
        raise ValueError("expected matching FP16 or E4M3FN inputs")
    if not a.is_cuda or not b.is_cuda or a.device != b.device:
        raise ValueError("expected the same CUDA device")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("expected rank-two tensors")
    M, K = a.shape
    K_b, N = b.shape
    if K != K_b:
        raise ValueError("incompatible dimensions")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("expected contiguous tensors")
    if a.is_neg() or b.is_neg():
        raise ValueError("expected resolved tensor storage")
    if M >= 2**63 or N >= 2**63 or K >= 2**63:
        raise ValueError("source signed-int64 metadata exceeded")
    dtype = a.dtype
    width = a.element_size()
    if M != 0 and N != 0:
        # Clamped source offsets and signed element-to-byte pointer increments.
        if max(width * M * K, width * K * N, width * M * N) >= 2**63:
            raise ValueError("source signed-int64 byte-offset envelope exceeded")
        if triton.cdiv(M, 128) * triton.cdiv(N, 256) >= 2**31:
            raise ValueError("source signed-int32 program-id envelope exceeded")
        for t in (a, b):
            if t.numel() != 0:
                start = t.data_ptr()
                if start % width != 0 or start + width * t.numel() > 2**64:
                    raise ValueError("input byte window is not aligned and representable")
    with torch.cuda.device(a.device):
        c = torch.empty((M, N), device=a.device, dtype=dtype)
        if M == 0 or N == 0:
            return c
        grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
        matmul_kernel_shifted[grid](
            a, b, c, M, N, K, K, 1, N, 1, max(N, 1), 1,
            BLOCK_SIZE_M=configs[dtype]["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=configs[dtype]["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=configs[dtype]["BLOCK_SIZE_K"],
            GROUP_SIZE_M=configs[dtype]["GROUP_SIZE_M"],
            num_stages=configs[dtype]["num_stages"],
            num_warps=configs[dtype]["num_warps"],
        )
    return c


matmul = matmul_shifted
