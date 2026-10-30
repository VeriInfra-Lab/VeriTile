import torch
import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def spinning_lock_kernel_area(P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, PHASE: tl.constexpr, P_NEG: tl.constexpr = False, C_NEG: tl.constexpr = False, LOCK_NEG: tl.constexpr = False):
    pid = tl.program_id(0).to(tl.int64)
    tl.static_assert(PHASE == 1 or PHASE == 2)
    is_reducer = pid.to(tl.uint64) % tl.cast(k, tl.uint64) == 0
    if (PHASE == 1 and is_reducer) or (PHASE == 2 and not is_reducer):
        return
    pid_m = pid // num_sms
    pid_n = pid % num_sms

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  # Assuming acc initialization

    lock_ready = -1 if LOCK_NEG else 1

    # Perform reduction for every kth pid
    for iters in range(1, 10):
        if is_reducer:
            next_pid = pid + 1

            while next_pid.to(tl.uint64) < pid.to(tl.uint64) + tl.cast(k, tl.uint64) and next_pid < num_sms:
                while tl.atomic_cas(locks + next_pid, lock_ready, lock_ready) != lock_ready:
                    pass

                rm1 = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
                rn1 = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
                P_ = P + next_pid * (BLOCK_SIZE_M * BLOCK_SIZE_N) + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc1 = tl.load(P_)
                if P_NEG:
                    acc1 = (acc1.to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.float32, bitcast=True)
                acc += acc1

                next_pid += 1
              
        # Store results using temporary storage P for every k-1 pids
        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
            rn1 = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
            P_ = P + pid * (BLOCK_SIZE_M * BLOCK_SIZE_N) + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, (acc.to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.float32, bitcast=True) if P_NEG else acc)
            tl.debug_barrier()
            tl.atomic_xchg(locks + pid, lock_ready)

        # Store final results in C
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, (acc.to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.float32, bitcast=True) if C_NEG else acc, mask=mask)


def _check_inputs(P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N):
    scalars = (num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N)
    if any(type(value) is not int for value in scalars):
        raise TypeError("dimensions, strides and launch parameters must be Python integers")
    if not (0 < num_sms < 1 << 31 and 0 < k < 1 << 64):
        raise ValueError("invalid program count or reduction group")
    if any(not (0 <= value < 1 << 63) for value in (M, N, stride_cm, stride_cn)):
        raise ValueError("dimensions and strides must fit nonnegative signed64")
    for block in (BLOCK_SIZE_M, BLOCK_SIZE_N):
        if block <= 0 or block > 1048576 or block & (block - 1):
            raise ValueError("block dimensions must be supported powers of two")
    lanes = BLOCK_SIZE_M * BLOCK_SIZE_N
    if lanes > 1048576:
        raise ValueError("tile exceeds the Triton block element bound")
    for tensor, dtype in ((P, torch.float32), (C, torch.float32), (locks, torch.int32)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("P, C and locks must be tensors")
        if tensor.layout != torch.strided or tensor.dtype != dtype:
            raise TypeError("unsupported tensor layout or dtype")
        if tensor.device.type != "cuda" or tensor.device != P.device:
            raise ValueError("all buffers must use the same CUDA device")
        if not tensor.is_contiguous() or tensor.is_neg() or tensor.is_conj():
            raise ValueError("buffers must be contiguous with ordinary view metadata")
    if C.shape != (M, N) or C.stride() != (stride_cm, stride_cn):
        raise ValueError("C shape and strides must equal the supplied metadata")
    if locks.dim() != 1 or P.numel() < num_sms * lanes or locks.numel() < num_sms:
        raise ValueError("partial or lock capacity is too small")
    if M and N and (M - 1) * stride_cm + (N - 1) * stride_cn >= C.numel():
        raise ValueError("output rectangle exceeds C")
    windows = []
    for tensor in (P, C, locks):
        base, size = tensor.data_ptr(), 4 * tensor.numel()
        if size >= 1 << 63 or base < 0 or base % 4 or base + size > 1 << 64:
            raise ValueError("buffer must be aligned and fit the native address domain")
        windows.append((base, base + size))
    for first, second in ((0, 1), (0, 2), (1, 2)):
        lo, hi = windows[first]
        other_lo, other_hi = windows[second]
        if not (hi <= other_lo or other_hi <= lo):
            raise ValueError("P, C and locks must occupy disjoint byte windows")
    formed = (BLOCK_SIZE_M - 1) * stride_cm + (num_sms * BLOCK_SIZE_N - 1) * stride_cn
    if formed >= 1 << 63 or C.data_ptr() + 4 * formed + 4 > 1 << 64:
        raise ValueError("masked output pointer arithmetic exceeds the native domain")
    # Validate the reusable publication invariant using exact words. Fresh
    # non-ready locks impose no restriction on the partial payloads.
    partial_words = P.detach().view(torch.int32).reshape(-1)
    for q, value in enumerate(locks.detach().cpu().tolist()):
        if value == 1:
            start, stop = q * lanes, (q + 1) * lanes
            if stop > P.numel() or bool((partial_words[start:stop] != 0).any().item()):
                raise ValueError("a ready lock must name a published positive-zero tile")


def spinning_lock_area(P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N):
    _check_inputs(P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N)
    grid = (num_sms,)
    p_neg = P.is_neg()
    c_neg = C.is_neg()
    lock_neg = locks.is_neg()
    # Same-stream launch ordering completes every producer before consumers run.
    with torch.cuda.device(P.device):
        for phase in (1, 2):
            spinning_lock_kernel_area[grid](
                P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
                PHASE=phase, P_NEG=p_neg, C_NEG=c_neg, LOCK_NEG=lock_neg,)
