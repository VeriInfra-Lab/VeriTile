import torch
import triton
from triton import language as tl

@triton.jit()
def swizzle_tile_area(tile_id,
                 M, N, K,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                 GROUP_M: tl.constexpr
                 ):
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = tile_id // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (tile_id % group_size)
    pid_n = (tile_id % width) // group_size
    return pid_m, pid_n


@triton.jit()
def linear_tile_area(tile_id,
                M, N, K,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                GROUP_M: tl.constexpr
                ):
    pid_m = tile_id // tl.cdiv(N, BLOCK_N)
    pid_n = tile_id % tl.cdiv(N, BLOCK_N)
    return pid_m, pid_n


@triton.jit()
def mac_loop_area(A, B, C,
             M, N, K,
             partials, program,
             stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
             iters_per_tile,
             start_iter, end_iter,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
             ACC_TYPE: tl.constexpr, GROUP_M: tl.constexpr):

    # where are we in the grid
    tile_id = start_iter // iters_per_tile
    if GROUP_M  > 0:
        pid_m, pid_n = swizzle_tile_area(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
    else:
        pid_m, pid_n = linear_tile_area(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K).to(tl.int64)
    step_a = tl.full((), BLOCK_K, tl.int64) * stride_ak
    step_b = tl.full((), BLOCK_K, tl.int64) * stride_bk
    A = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak) + step_a * (start_iter % iters_per_tile)
    B = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn) + step_b * (start_iter % iters_per_tile)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for current_iter in range(start_iter, end_iter):
        kk = (current_iter % iters_per_tile) * BLOCK_K + rk
        a = tl.load(A, mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(B, mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        A += step_a
        B += step_b

    if end_iter % iters_per_tile == 0:  # unique terminal chunk initializes this tile
        C_ = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)  # compute inside the if/else to avoid spilling!
        tl.store(C_, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))
    else:
        # A program has at most one nonterminal chunk: its final chunk.
        slots = program * (BLOCK_M * BLOCK_N) + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        tl.store(partials + slots, acc)


@triton.jit()
def first_wave_area(
        A, B, C,
        M, N, K,
        partials,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        total_full_tiles_streamk, total_partial_tiles_streamk, iters_per_tile,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ACC_TYPE: tl.constexpr,
        GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    start_iter = pid * total_full_tiles_streamk + tl.minimum(pid, total_partial_tiles_streamk)
    last_iter = (pid + 1) * total_full_tiles_streamk + tl.minimum(pid + 1, total_partial_tiles_streamk)

    while start_iter < last_iter:
        end_iter = tl.minimum(start_iter + (iters_per_tile - start_iter % iters_per_tile), last_iter)
        mac_loop_area(A, B, C,
                 M, N, K,
                 partials, pid,
                 stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                 iters_per_tile,
                 start_iter, end_iter,
                 BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE,
                 GROUP_M,
                 )

        start_iter = end_iter


@triton.jit()
def full_tiles_area(
        A, B, C,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        total_tiles_streamk,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ACC_TYPE: tl.constexpr,
        GROUP_M: tl.constexpr,
):
    # first wave has done more tiles than there are SMs, we adjust pid
    tile_id = tl.program_id(0).to(tl.int64) + total_tiles_streamk
    if GROUP_M > 0:
        pid_m, pid_n = swizzle_tile_area(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
    else:
        pid_m, pid_n = linear_tile_area(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)

    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K).to(tl.int64)
    # Widen pointer increments before multiplying by the runtime strides.
    step_a = tl.full((), BLOCK_K, tl.int64) * stride_ak
    step_b = tl.full((), BLOCK_K, tl.int64) * stride_bk
    # pointers
    A = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        kk = (tl.full((), 0, tl.int64) + k) * BLOCK_K + rk
        a = tl.load(A, mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(B, mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        A += step_a
        B += step_b
    acc = acc.to(tl.float16)  # restore C.dtype.element_ty
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(C, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))



@triton.jit()
def reduce_streamk_area(C, partials, M, N, K, stride_cm, stride_cn,
                   total_programs_streamk, total_full_tiles_streamk,
                   total_partial_tiles_streamk, iters_per_tile,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    tile_id = tl.program_id(0).to(tl.int64)
    if GROUP_M > 0:
        pid_m, pid_n = swizzle_tile_area(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
    else:
        pid_m, pid_n = linear_tile_area(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    for program in range(0, total_programs_streamk):
        pid = tl.full((), 0, tl.int64) + program
        begin = pid * total_full_tiles_streamk + tl.minimum(pid, total_partial_tiles_streamk)
        end = (pid + 1) * total_full_tiles_streamk + tl.minimum(pid + 1, total_partial_tiles_streamk)
        if (begin < end) & (end % iters_per_tile != 0) & (end // iters_per_tile == tile_id):
            slots = pid * (BLOCK_M * BLOCK_N) + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
            contribution = tl.load(partials + slots)
            tl.atomic_add(C + offsets, contribution, mask=mask)


class matmul_area(torch.autograd.Function):

    _debug = False

    @staticmethod
    def set_debug(debug: bool):
        matmul_area._debug = debug

    @staticmethod
    def _call(a: torch.Tensor, b: torch.Tensor, total_programs_streamk: int, BLK_M: int, BLK_N: int, BLK_K: int, two_tiles: bool, num_stages: int, num_warps: int):
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise TypeError("expected Tensor arguments")
        if a.layout != torch.strided or b.layout != torch.strided or a.ndim != 2 or b.ndim != 2:
            raise ValueError("expected rank-two strided tensors")
        if a.dtype != torch.float32 or b.dtype != torch.float32:
            raise ValueError("expected the original FP32 storage profile")
        if not a.is_cuda or not b.is_cuda or a.device != b.device:
            raise ValueError("expected inputs on the same CUDA device")
        if not a.is_contiguous() or not b.is_contiguous() or a.is_neg() or b.is_neg():
            raise ValueError("expected contiguous, resolved tensor storage")
        if any(not isinstance(x, int) or isinstance(x, bool) for x in
               (total_programs_streamk, BLK_M, BLK_N, BLK_K, num_stages, num_warps)):
            raise TypeError("expected integer launch parameters")
        if not isinstance(two_tiles, bool):
            raise TypeError("expected a Boolean two_tiles selector")
        if any(x <= 0 or x > 2**20 or x & (x - 1) for x in (BLK_M, BLK_N, BLK_K)) or BLK_K < 8 or BLK_K % 8 != 0:
            raise ValueError("expected supported power-of-two block widths")
        if num_stages <= 0 or num_warps not in (1, 2, 4, 8, 16, 32):
            raise ValueError("unsupported pipeline or warp configuration")
        device = a.device

        if a.shape[1] != b.shape[0]:
            raise ValueError("incompatible dimensions")
        M, K = a.shape
        _, N = b.shape
        # accumulator types
        ACC_TYPE = tl.float32 if a.dtype in [torch.float16, torch.bfloat16, torch.float32] else tl.int32
        # compute grid (work to do per SM on the first wave)
        total_blocks_M = triton.cdiv(M, BLK_M)
        total_blocks_N = triton.cdiv(N, BLK_N)
        iters_per_tile = triton.cdiv(K, BLK_K)
        GROUP_M = 8  # 0 to disable swizzling
        total_tiles = total_blocks_M * total_blocks_N

        if total_programs_streamk > 0 and K > 0:  # Stream-K
            # last wave may occupy less than total_programs_streamk SMs
            total_tiles_streamk = total_tiles % total_programs_streamk
            # for two-tile Stream-K + data-parallel from original paper
            if two_tiles and total_tiles - total_tiles_streamk > total_programs_streamk:
                total_tiles_streamk += total_programs_streamk
            # remaining tiles are computed using classical blocking
            total_blocking_tiles = total_tiles - total_tiles_streamk
            total_iters_streamk = total_tiles_streamk * iters_per_tile
            # iterations related to full waves
            total_full_tiles_streamk = total_iters_streamk // total_programs_streamk
            # iterations related to last (partial) wave
            total_partial_tiles_streamk = total_iters_streamk % total_programs_streamk

        else:  # all tiles are computed using classical blocking
            total_blocking_tiles = total_tiles
            total_tiles_streamk = 0
            total_full_tiles_streamk = 0
            total_partial_tiles_streamk = 0
            total_iters_streamk = 0

        if M != 0 and N != 0:
            # Source-static Triton constraints, also when runtime K is zero.
            if BLK_K < 16:
                raise ValueError("FP32 dot requires BLOCK_K >= 16")
            if max(BLK_M * BLK_K, BLK_K * BLK_N, BLK_M * BLK_N) > 2**20:
                raise ValueError("source block tensor exceeds 2**20 elements")
            # Source i32 ceilings/group width and signed program-id ABI.
            if max(M + BLK_M - 1, N + BLK_N - 1, K + BLK_K - 1,
                   8 * total_blocks_N, total_tiles, max(total_programs_streamk, 0)) >= 2**31:
                raise ValueError("source signed-int32 control envelope exceeded")
            if total_iters_streamk + iters_per_tile >= 2**63:
                raise ValueError("source signed-int64 stream boundary exceeded")
            # Include terminal pointer advances and inactive tile-tail pointers.
            padded_m, padded_n = total_blocks_M * BLK_M, total_blocks_N * BLK_N
            advanced_k = (iters_per_tile + 1) * BLK_K
            pointer_a = padded_m * K + advanced_k
            pointer_b = advanced_k * N + padded_n
            pointer_c = padded_m * max(N, 1) + padded_n
            scratch = max(total_programs_streamk, 0) * BLK_M * BLK_N
            if 4 * max(pointer_a, pointer_b, pointer_c, scratch, M * K, K * N, M * N) >= 2**63:
                raise ValueError("source signed-int64 byte-offset envelope exceeded")
            for tensor in (a, b):
                if tensor.numel() != 0:
                    ptr = tensor.data_ptr()
                    if ptr % 4 != 0 or ptr + 4 * tensor.numel() > 2**64:
                        raise ValueError("input byte window is not aligned and representable")

        if matmul_area._debug:
            print(f"M,N,K={M},{N},{K} ; BLK_M,N,K={BLK_M},{BLK_N},{BLK_K}")
            print(f"{total_blocks_M=} x {total_blocks_N=} = {total_tiles=}")
            print(f"{total_tiles_streamk=} + {total_blocking_tiles=} = {total_tiles=}")
            print(f"{total_programs_streamk=}")
            print(f"{total_blocking_tiles=}")
            print(f"{iters_per_tile=}")
            print(f"{total_iters_streamk=}")

        # allocates output
        c = torch.empty((M, N), device=device, dtype=a.dtype)
        if M == 0 or N == 0:
            return c
        if total_tiles_streamk > 0:
            partials = torch.empty((total_programs_streamk, BLK_M, BLK_N), device=device, dtype=c.dtype)
            k1 = first_wave_area[(total_programs_streamk,)](
                a,
                b,
                c,
                M,
                N,
                K,
                partials,
                K, 1, N, 1, max(N, 1), 1,
                total_full_tiles_streamk=total_full_tiles_streamk,
                total_partial_tiles_streamk=total_partial_tiles_streamk,
                iters_per_tile=iters_per_tile,
                BLOCK_M=BLK_M,
                BLOCK_N=BLK_N,
                BLOCK_K=BLK_K,
                ACC_TYPE=ACC_TYPE,
                GROUP_M=GROUP_M,
                num_stages=num_stages,
                num_warps=num_warps,
            )
            if matmul_area._debug:
                print(f"{k1.n_regs} registers used, {k1.n_spills} spills")
            reduce_streamk_area[(total_tiles_streamk,)](
                c, partials, M, N, K, max(N, 1), 1,
                total_programs_streamk, total_full_tiles_streamk,
                total_partial_tiles_streamk, iters_per_tile,
                BLOCK_M=BLK_M, BLOCK_N=BLK_N, BLOCK_K=BLK_K,
                GROUP_M=GROUP_M, num_stages=num_stages, num_warps=num_warps,
            )
        if total_blocking_tiles > 0:
            k2 = full_tiles_area[(total_blocking_tiles,)](
                a,
                b,
                c,
                M,
                N,
                K,
                K, 1, N, 1, max(N, 1), 1,
                total_tiles_streamk=total_tiles_streamk,
                BLOCK_M=BLK_M,
                BLOCK_N=BLK_N,
                BLOCK_K=BLK_K,
                ACC_TYPE=ACC_TYPE,
                GROUP_M=GROUP_M,
                num_stages=num_stages,
                num_warps=num_warps,
            )
            if matmul_area._debug:
                print(f"{k2.n_regs} registers used, {k2.n_spills} spills")
        return c

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, grid: int, BLK_M=128, BLK_N=128, BLK_K=32, two_tiles=True, num_stages=3, num_warps=4):
        return matmul_area._call(a=a, b=b, total_programs_streamk=grid, BLK_M=BLK_M, BLK_N=BLK_N, BLK_K=BLK_K, two_tiles=two_tiles, num_warps=num_warps, num_stages=num_stages)

