import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    m_size,
    k_size,
    n_size,
    stride_xm, stride_xk, stride_yk, stride_yn, stride_zm, stride_zn,
    m_block_size: tl.constexpr,
    k_block_size: tl.constexpr,
    n_block_size: tl.constexpr,
):
    m_size = tl.cast(m_size, tl.uint64)
    k_size = tl.cast(k_size, tl.uint64)
    n_size = tl.cast(n_size, tl.uint64)
    stride_xm = tl.cast(stride_xm, tl.uint64)
    stride_xk = tl.cast(stride_xk, tl.uint64)
    stride_yk = tl.cast(stride_yk, tl.uint64)
    stride_yn = tl.cast(stride_yn, tl.uint64)
    stride_zm = tl.cast(stride_zm, tl.uint64)
    stride_zn = tl.cast(stride_zn, tl.uint64)
    pid = tl.program_id(0).to(tl.uint64)
    num_n_blocks = tl.cdiv(n_size, n_block_size)
    m_block = pid // num_n_blocks
    n_block = pid % num_n_blocks

    m_offsets = tl.arange(0, m_block_size).to(tl.uint64) + m_block * m_block_size
    n_offsets = tl.arange(0, n_block_size).to(tl.uint64) + n_block * n_block_size
    k_offsets = tl.arange(0, k_block_size).to(tl.uint64)

    row_mask = m_offsets < m_size
    column_mask = n_offsets < n_size

    x_ptrs = x_ptr + m_offsets[:, None] * stride_xm + k_offsets[None, :] * stride_xk
    y_ptrs = y_ptr + k_offsets[:, None] * stride_yk + n_offsets[None, :] * stride_yn
    z_ptrs = z_ptr + m_offsets[:, None] * stride_zm + n_offsets[None, :] * stride_zn

    z = tl.zeros((m_block_size, n_block_size), dtype=tl.float32)

    for k in range(0, tl.cdiv(k_size, k_block_size)):
        x_sub = tl.load(x_ptrs, mask=row_mask[:, None] & (k_offsets[None, :] < k_size - k * k_block_size), other=0.0)
        y_sub = tl.load(y_ptrs, mask=(k_offsets[:, None] < k_size - k * k_block_size) & column_mask[None, :], other=0.0)
        z += tl.dot(x_sub, y_sub, allow_tf32=False)
        x_ptrs += k_block_size * stride_xk
        y_ptrs += k_block_size * stride_yk

    tl.store(z_ptrs, z, mask=row_mask[:, None] & column_mask[None, :])

def matmul(x, y):
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("expected Tensor inputs")
    if x.layout != torch.strided or y.layout != torch.strided:
        raise ValueError("expected strided inputs")
    if x.dtype != torch.float32 or y.dtype != torch.float32:
        raise ValueError("expected FP32 inputs")
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("expected CUDA inputs")
    if x.device != y.device:
        raise ValueError("expected the same device")
    if x.device.index != 0:
        raise ValueError("expected cuda:0 inputs")
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("expected FP32 default output dtype")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("expected matrices")
    if x.shape[1] != y.shape[0]:
        raise ValueError("incompatible dimensions")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("expected contiguous inputs")
    if x.is_neg() or y.is_neg():
        raise ValueError("expected resolved input storage")
    m_size, k_size = x.shape
    _, n_size = y.shape
    if m_size >= 2**63 or k_size >= 2**63 or n_size >= 2**63:
        raise ValueError("dimensions must fit signed 64-bit metadata")
    if (x.numel() > 0 and x.data_ptr() % 4 != 0) or (y.numel() > 0 and y.data_ptr() % 4 != 0):
        raise ValueError("nonempty FP32 inputs require aligned pointers")
    z = torch.empty(m_size, n_size, device='cuda:0')
    if m_size == 0 or n_size == 0:
        return z
    def grid(meta):
        return (triton.cdiv(m_size, meta['m_block_size']) * triton.cdiv(n_size, meta['n_block_size']),)

    with torch.cuda.device(x.device):
        matmul_kernel[grid](
            x,
            y,
            z,
            m_size,
            k_size,
            n_size,
            k_size, 1,
            n_size, 1,
            n_size, 1,
            32,
            32,
            32
        )
    return z
