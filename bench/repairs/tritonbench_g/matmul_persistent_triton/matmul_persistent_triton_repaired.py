
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
def matmul_kernel_persistent(a_ptr, b_ptr, c_ptr,  #
                             M, N, K,  #
                             stride_am, stride_ak,  #
                             stride_bk, stride_bn,  #
                             stride_cm, stride_cn,  #
                             BLOCK_SIZE_M: tl.constexpr,  #
                             BLOCK_SIZE_N: tl.constexpr,  #
                             BLOCK_SIZE_K: tl.constexpr,  #
                             GROUP_SIZE_M: tl.constexpr,  #
                             NUM_SMS: tl.constexpr,  #
                             ):
    start_pid = tl.program_id(axis=0).to(tl.int64)
    M, N, K = M.to(tl.int64), N.to(tl.int64), K.to(tl.int64)
    stride_am, stride_ak = stride_am.to(tl.int64), stride_ak.to(tl.int64)
    stride_bk, stride_bn = stride_bk.to(tl.int64), stride_bn.to(tl.int64)
    stride_cm, stride_cn = stride_cm.to(tl.int64), stride_cn.to(tl.int64)
    num_pid_m = M // BLOCK_SIZE_M + (M % BLOCK_SIZE_M != 0)
    num_pid_n = N // BLOCK_SIZE_N + (N % BLOCK_SIZE_N != 0)
    k_tiles = K // BLOCK_SIZE_K + (K % BLOCK_SIZE_K != 0)
    num_tiles = num_pid_m * num_pid_n
    tiles_per_SM = num_tiles // NUM_SMS + (start_pid < num_tiles % NUM_SMS)
    active_group_m = min(GROUP_SIZE_M, num_pid_m)
    num_pid_in_group = active_group_m * num_pid_n
    lanes_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    lanes_n = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

    for tile_ordinal in range(0, tiles_per_SM):
        tile_id = start_pid + tile_ordinal * NUM_SMS
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * active_group_m
        group_size_m = min(num_pid_m - first_pid_m, active_group_m)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_cm = start_m + lanes_m
        offs_cn = start_n + lanes_n
        offs_am = tl.where(offs_cm < M, offs_cm, 0)
        offs_bn = tl.where(offs_cn < N, offs_cn, 0)
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for ki in range(0, k_tiles):
            raw_k = ki * BLOCK_SIZE_K + offs_k_for_mask
            offs_k = tl.where(raw_k < K, raw_k, 0)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)

        c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if (c_ptr.dtype.element_ty == tl.float8e4nv):
            c = accumulator.to(tl.float8e4nv)
        else:
            c = accumulator.to(tl.float16)
        tl.store(c_ptrs, c, mask=c_mask)


def matmul_persistent(a, b):
    configs = {
        torch.float8_e4m3fn: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_stages": 4,
            "num_warps": 8
        }, torch.float16: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_stages": 3,
            "num_warps": 8
        }
    }
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    config = configs[dtype]
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    if M == 0 or N == 0:
        return c
    if K == 0:
        c.zero_()
        return c
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    with torch.cuda.device(a.device):
        matmul_kernel_persistent[grid](
            a, b, c,  #
            M, N, K,  #
            a.stride(0), a.stride(1),  #
            b.stride(0), b.stride(1),  #
            c.stride(0), c.stride(1),  #
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],  #
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],  #
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],  #
            GROUP_SIZE_M=config["GROUP_SIZE_M"],  #
            NUM_SMS=NUM_SMS,  #
            num_stages=config["num_stages"],  #
            num_warps=config["num_warps"],  #
        )
    return c

_unchecked_matmul_persistent = matmul_persistent


def matmul_persistent(a, b):
    """Checked raw FP16/E4M3 inputs for the proved persistent source."""
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("a and b must be tensors")
    if a.layout != torch.strided or b.layout != torch.strided:
        raise ValueError("a and b must have strided layout")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must have rank two")
    if a.dtype not in (torch.float16, torch.float8_e4m3fn) or b.dtype != a.dtype:
        raise TypeError("a and b must have matching FP16 or E4M3 dtype")
    if not a.is_cuda or not b.is_cuda or a.device != b.device:
        raise ValueError("a and b must share a CUDA device")
    if a.shape[1] != b.shape[0]:
        raise ValueError("incompatible contraction dimensions")
    if a.is_neg() or b.is_neg() or a.is_conj() or b.is_conj():
        raise ValueError("unresolved tensor views are unsupported")
    if min(*a.stride(), *b.stride()) < 0:
        raise ValueError("negative strides are unsupported")
    if any(x.numel() and x.data_ptr() % x.element_size() for x in (a, b)):
        raise ValueError("nonempty input pointers must have natural element alignment")
    return _unchecked_matmul_persistent(a, b)
