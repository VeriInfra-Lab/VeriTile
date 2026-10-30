import triton
import triton.language as tl
import torch

MAX_FUSED_SIZE = 65536  # 2**16
next_power_of_2 = triton.next_power_of_2

def calculate_settings(n):
    BLOCK_SIZE = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "\
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps = 4
    if   BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >=  8192: num_warps = 16
    elif BLOCK_SIZE >=  2048: num_warps = 8
    return BLOCK_SIZE, num_warps


ROPE_GROUP_SIZE = 4

@triton.heuristics({"BACKWARD_PASS": lambda args: args["BACKWARD_PASS"],})
@triton.jit
def _rope_embedding(
    Q,     Q_row_stride,
    cos, cos_row_stride,
    sin, sin_row_stride,
    seqlen,
    head_dim        : tl.constexpr,
    n_heads         : tl.constexpr,
    BACKWARD_PASS   : tl.constexpr,
    BLOCK_SIZE      : tl.constexpr,
    ROPE_GROUP_SIZE : tl.constexpr = 4,
):
    """
        Calculates the RoPE Embedding quickly
        RoPE is Q * cos + rotate_half(Q) * sin
        See our blog post for more info
    """
    row_position  = tl.program_id(0)
    group_head_position = tl.program_id(1)
    col_offsets  = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim

    sin1 = tl.load(sin + (row_position % seqlen)*sin_row_stride + \
                   half_head_dim*0 + col_offsets, mask = mask, other = 0)
    cos1 = tl.load(cos + (row_position % seqlen)*cos_row_stride + \
                   half_head_dim*0 + col_offsets, mask = mask, other = 0)

    if BACKWARD_PASS:
        # See our blog post for more info.
        sin1 = -sin1
    
    # [TODO] Autotune ROPE_GROUP_SIZE to be 1, 2, 4, 8
    head_start = group_head_position * ROPE_GROUP_SIZE
    head_end = min((head_start + ROPE_GROUP_SIZE), n_heads)

    # 10% Faster kernel from [HuyNguyen-hust](https://github.com/unslothai/unsloth/pull/238)
    for k in range(head_start, head_end):
        offs_q1 = row_position * Q_row_stride + k * head_dim + col_offsets
        offs_q2 = row_position * Q_row_stride + k * head_dim + col_offsets + half_head_dim

        # For Gemma - sometimes RoPE must be done in float32 and not bfloat16
        Q1 = tl.load(Q + offs_q1, mask = mask, other = 0).to(sin1.dtype)
        Q2 = tl.load(Q + offs_q2, mask = mask, other = 0).to(sin1.dtype)

        tl.store(Q + offs_q1, Q1*cos1 - Q2*sin1, mask = mask)
        tl.store(Q + offs_q2, Q2*cos1 + Q1*sin1, mask = mask)
    


def _check_payload(Q, cos, sin):
    tensors = (Q, cos, sin)
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("expected Tensor arguments")
    if any(t.layout != torch.strided for t in tensors):
        raise ValueError("only strided tensors are supported")
    if any(t.dtype != torch.float32 for t in tensors):
        raise TypeError("the checked profile requires FP32")
    if any(not t.is_cuda or t.device != Q.device for t in tensors):
        raise ValueError("all tensors must share one CUDA device")
    if any(not t.is_contiguous() or t.is_neg() or t.is_conj() for t in tensors):
        raise ValueError("contiguous resolved tensors required")
    if Q.ndim != 4:
        raise ValueError("input must have shape [B,H,S,D]")
    for t in tensors:
        storage = t.untyped_storage()
        if t.numel() and not (storage.data_ptr() <= t.data_ptr()
                and t.data_ptr() + 4 * t.numel() <= storage.data_ptr() + storage.nbytes() <= 2**64
                and t.data_ptr() % 4 == 0):
            raise ValueError("FP32 view must fit aligned backing storage")


def _check_tables(cos, sin, sequence, half):
    for t in (cos, sin):
        if not (t.ndim == 2 and t.shape[1] == half
                or half == 1 and t.ndim == 1):
            raise ValueError("source coefficient views must be [L,D//2] or [L] when D//2=1")
        if t.shape[0] < sequence or (half and t.stride(0) != half):
            raise ValueError("coefficient views must cover the sequence with packed rows")


def _check_alias(Q, cos, sin):
    a = Q.untyped_storage()
    for t in (cos, sin):
        b = t.untyped_storage()
        if a.nbytes() and b.nbytes() and not (
                a.data_ptr() + a.nbytes() <= b.data_ptr()
                or b.data_ptr() + b.nbytes() <= a.data_ptr()):
            raise ValueError("reshape input and coefficient backing allocations must be separate")


def _geometry(batch, heads, sequence, dimension):
    half = dimension // 2
    block, warps = calculate_settings(half) if half else (0, 4)
    groups = (heads + 3) // 4
    rows = batch * sequence
    if max(batch, heads + 3, sequence, dimension, rows, heads * dimension, groups) >= 2**31:
        raise ValueError("launch metadata exceeds signed-i32 profile")
    if rows and heads and half:
        output_last = ((rows - 1) * heads * dimension + (heads - 1) * dimension
                       + block - 1 + half)
        table_last = (sequence - 1) * half + block - 1
        if max(output_last, table_last) >= 2**31:
            raise ValueError("formed padded offsets exceed signed-i32 profile")
    else:
        output_last = table_last = 0
    return groups, block, warps, output_last, table_last


def _check_pointers(Q, cos, sin, output_last, table_last):
    if any(t.data_ptr() + 4 * last + 4 > 2**64
           for t, last in ((Q, output_last), (cos, table_last), (sin, table_last))):
        raise ValueError("formed pointers exceed 64-bit profile")


def _rope_embedding_forward_impl(Q, cos, sin):
    """Checked original clone/reshape forward, returning actual squeeze views."""
    _check_payload(Q, cos, sin)
    batch, n_heads, seq_len, head_dim = Q.shape
    n_groups, BLOCK_SIZE, num_warps, output_last, table_last = _geometry(batch, n_heads, seq_len, head_dim)
    with torch.cuda.device(Q.device):
        Q = Q.transpose(1, 2).clone()
        cos, sin = cos.squeeze(), sin.squeeze()
        _check_tables(cos, sin, seq_len, head_dim // 2)
        Q = Q.reshape(batch * seq_len, n_heads * head_dim)
        if Q.numel() and head_dim >= 2:
            _check_pointers(Q, cos, sin, output_last, table_last)
            _rope_embedding[(batch * seq_len, n_groups)](
                Q, Q.stride(0), cos, cos.stride(0), sin, sin.stride(0),
                seq_len, head_dim, n_heads, BACKWARD_PASS=False,
                BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
            )
        return Q.view(batch, seq_len, n_heads, head_dim).transpose(1, 2), cos, sin, n_groups, BLOCK_SIZE, num_warps


def _rope_embedding_backward_impl(dY, cos, sin, n_groups, BLOCK_SIZE, num_warps):
    """Checked original reshape backward; incoming gradient storage may mutate."""
    _check_payload(dY, cos, sin)
    batch, n_heads, seq_len, head_dim = dY.shape
    groups, block, warps, output_last, table_last = _geometry(batch, n_heads, seq_len, head_dim)
    if (n_groups, BLOCK_SIZE, num_warps) != (groups, block, warps):
        raise ValueError("cached launch geometry must match the actual gradient shape")
    _check_tables(cos, sin, seq_len, head_dim // 2)
    _check_alias(dY, cos, sin)
    with torch.cuda.device(dY.device):
        dY = dY.transpose(1, 2)
        dY = dY.reshape(batch * seq_len, n_heads * head_dim)
        if dY.numel() and head_dim >= 2:
            _check_pointers(dY, cos, sin, output_last, table_last)
            _rope_embedding[(batch * seq_len, n_groups)](
                dY, dY.stride(0), cos, cos.stride(0), sin, sin.stride(0),
                seq_len, head_dim, n_heads, BACKWARD_PASS=True,
                BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
            )
        return dY.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
