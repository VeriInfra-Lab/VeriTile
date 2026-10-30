import torch

import triton
import triton.language as tl


@triton.jit
def matmul_kernel_with_block_pointers(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See the matrix multiplication tutorial for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    GROUP_M = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % GROUP_M)
    pid_n = (pid % num_pid_in_group) // GROUP_M

    # ----------------------------------------------------------
    # Create block pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction and accumulate.
    # See above `Make a Block Pointer` section for details.
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_M, BLOCK_N]` block.
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, K, BLOCK_K):
        # Load with boundary checks, no need to calculate the mask manually.
        # For better performance, you may remove some axis from the boundary
        # check, if you can guarantee that the access is always in-bound in
        # that axis.
        # See above `Load/Store a Block Pointer` section for details.
        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")
        # We accumulate along the K dimension.
        # Keep the wrapping add separate from Triton's saturating INT8 MMA.
        accumulator = tl.inline_asm_elementwise(
            "add.u32 $0, $1, $2;", constraints="=r,r,r",
            args=[accumulator, tl.dot(a, b)], dtype=tl.int32, is_pure=True, pack=1,
        )
        # Advance the block pointer to the next K block.
        # See above `Advance a Block Pointer` section for details.
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
    c = accumulator  # .to(tl.float16)

    # ----------------------------------------------------------------
    # Write back the block of the output matrix C with boundary checks.
    # See above `Load/Store a Block Pointer` section for details.
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


@triton.jit
def scaled_matmul_kernel_with_block_pointers(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    s1_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_s1m,
    stride_s1n,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    ACC_TYPE: tl.constexpr = tl.int32,
):
    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = rm % M
    rbn = rn % N
    rk = tl.arange(0, BLOCK_K)
    A = a_ptr + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = b_ptr + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        # A zero-seeded tile dot is bounded; the carry update wraps modulo 2**32.
        acc = tl.inline_asm_elementwise(
            "add.u32 $0, $1, $2;", constraints="=r,r,r",
            args=[acc, tl.dot(a, b)], dtype=tl.int32, is_pure=True, pack=1,
        )
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n * stride_cn + idx_m * stride_cm
    tmp0 = tl.load(
        s1_ptr + (tl.broadcast_to(idx_m * stride_s1m, mask.shape)),
        mask,
        eviction_policy="evict_last",
    )
    tl.store(c_ptr + (tl.broadcast_to(xindex, mask.shape)), acc * tmp0, mask)


def _checked_inputs(a, b, c, config, scales1=None, *, scaled=False):
    tensors = (a, b, c) if not scaled else (a, b, c, scales1)
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("expected Tensor arguments")
    if any(t.layout != torch.strided for t in tensors):
        raise ValueError("expected strided tensors")
    if a.dtype != torch.int8 or b.dtype != torch.int8 or c.dtype != torch.int32:
        raise ValueError("expected INT8 inputs and INT32 output")
    if scaled and scales1.dtype != torch.float32:
        raise ValueError("expected FP32 row scales")
    if any(not t.is_cuda for t in tensors):
        raise ValueError("expected CUDA tensors")
    if any(t.device != a.device for t in tensors):
        raise ValueError("expected the same CUDA device")
    if any(t.ndim != 2 for t in tensors):
        raise ValueError("expected rank-two tensors")
    M, K = a.shape
    K_b, N = b.shape
    if K != K_b or tuple(c.shape) != (M, N):
        raise ValueError("incompatible matmul shapes")
    if scaled and tuple(scales1.shape) != (M, 1):
        raise ValueError("expected one scale per output row")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("expected contiguous tensors")
    if any(t.is_neg() for t in tensors):
        raise ValueError("expected resolved tensor storage")
    if not isinstance(config, Config):
        raise TypeError("expected Config")
    if set(config.kwargs) != {"BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M"}:
        raise ValueError("expected the four matrix tile parameters")
    BM, BN, BK, G = (config.kwargs[k] for k in ("BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M"))
    if any(type(v) is not int for v in (BM, BN, BK, G, config.num_warps, config.num_stages, config.num_ctas)):
        raise TypeError("expected builtin integer configuration values")
    if any(v < minimum or v & (v - 1) for v, minimum in ((BM, 16), (BN, 16), (BK, 32))):
        raise ValueError("expected legal power-of-two INT8 dot tiles")
    if max(BM * BN, BM * BK, BK * BN) > 2**20 or G <= 0:
        raise ValueError("tile or group exceeds the supported Triton shape")
    if config.num_warps not in (1, 2, 4, 8, 16, 32) or config.num_stages <= 0 or config.num_ctas != 1:
        raise ValueError("unsupported launch configuration")
    if M == 0 or N == 0:
        return M, N, K
    gm, gn, gk = triton.cdiv(M, BM), triton.cdiv(N, BN), triton.cdiv(K, BK)
    pm, pn, pk = gm * BM, gn * BN, gk * BK
    # Bound the actual cdiv numerators, grouped mapping, padded tile offsets,
    # and pointer values after the final K advance in signed-int32 arithmetic.
    envelope = (M + BM - 1, N + BN - 1, K + BK - 1, G * gn, gm * gn,
                pm * K + pk + BK, (pk + BK) * N + pn, pm * N + pn)
    if any(v >= 2**31 for v in envelope):
        raise ValueError("source signed-int32 index envelope exceeded")
    windows = []
    for t in tensors:
        if t.numel() == 0:
            windows.append((0, 0))
            continue
        start = t.data_ptr()
        width = t.element_size()
        end = start + width * t.numel()
        if start % width != 0 or end > 2**64:
            raise ValueError("tensor byte window is not aligned and representable")
        windows.append((start, end))
    c_start, c_end = windows[2]
    for i in (0, 1) if not scaled else (0, 1, 3):
        start, end = windows[i]
        if start < end and c_start < end and start < c_end:
            raise ValueError("output must not overlap inputs or scales")
    return M, N, K


def int_matmul_kernel(a, b, c, config):
    M, N, K = _checked_inputs(a, b, c, config)
    if M == 0 or N == 0:
        return c
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    with torch.cuda.device(a.device):
        matmul_kernel_with_block_pointers[grid](
            a, b, c, M, N, K, K, 1, N, 1, N, 1,
            num_warps=config.num_warps, num_stages=config.num_stages,
            num_ctas=config.num_ctas, **config.kwargs,
        )
    return c


def int_scaled_matmul_kernel(a, b, scales1, c, config):
    M, N, K = _checked_inputs(a, b, c, config, scales1, scaled=True)
    if M == 0 or N == 0:
        return c
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    with torch.cuda.device(a.device):
        scaled_matmul_kernel_with_block_pointers[grid](
            a, b, c, scales1, M, N, K, K, 1, N, 1, N, 1, 1, 1,
            num_warps=config.num_warps, num_stages=config.num_stages,
            num_ctas=config.num_ctas,
            EVEN_K=(K % config.kwargs["BLOCK_K"] == 0), **config.kwargs,
        )
    return c


class Config:
    def __init__(self, num_warps=4, num_stages=2, num_ctas=1, **kwargs):
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.num_ctas = num_ctas
        self.kwargs = kwargs




