import torch
import triton
import triton.language as tl

@triton.jit
def diag_ssm_forward_kernel(s_ptr, x_ptr, lambda_ptr, y_ptr, length,
                            batch_size, dim, BLOCK_SIZE: tl.constexpr):
    """
    Forward kernel (real version)

    Arguments:
        s_ptr: [batch_size, dim]
        x_ptr: [length, batch_size, dim]
        lambda_ptr: [dim]
        y_ptr: [length, batch_size, dim]
    """
    # Widen before every shape product, time loop and pointer offset.
    length = tl.cast(length, tl.int64)
    batch_size = tl.cast(batch_size, tl.int64)
    dim = tl.cast(dim, tl.int64)
    col_idx = tl.cast(tl.program_id(0), tl.int64) * BLOCK_SIZE
    col_offsets = col_idx + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < batch_size * dim
    s = tl.load(s_ptr + col_offsets, mask=mask, other=0)
    Lambda = tl.load(lambda_ptr + col_offsets % dim, mask=mask, other=0)
    for t in range(length):
        offsets = t * batch_size * dim + col_offsets
        x = tl.load(x_ptr + offsets, mask=mask, other=0)
        s = s * Lambda + x
        tl.store(y_ptr + offsets, s, mask=mask)

@triton.jit
def diag_ssm_backward_kernel(
        s_ptr, lambda_ptr, y_ptr, grad_s_ptr, grad_x_ptr, grad_lambda_ptr,
        grad_y_ptr, length, batch_size, dim, BLOCK_SIZE: tl.constexpr):
    """
    Backward kernel (real version)

    Arguments:
        s_ptr: [batch_size, dim]
        lambda_ptr: [dim]
        y_ptr: [length, batch_size, dim]
        grad_s_ptr: [batch_size, dim]
        grad_x_ptr: [length, batch_size, dim]
        grad_lambda_ptr: [batch_size, dim]
        grad_y_ptr: [length, batch_size, dim]
    """

    # Widen before every shape product, time loop and pointer offset.
    length = tl.cast(length, tl.int64)
    batch_size = tl.cast(batch_size, tl.int64)
    dim = tl.cast(dim, tl.int64)
    col_idx = tl.cast(tl.program_id(0), tl.int64) * BLOCK_SIZE
    col_offsets = col_idx + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < batch_size * dim

    Lambda = tl.load(lambda_ptr + col_offsets % dim, mask=mask, other=0)

    # Initialize gradients to zero.
    grad_s = tl.zeros_like(Lambda)
    grad_Lambda = tl.zeros_like(Lambda)

    for i in range(length):
        # Triton does not support range(length - 1, -1, -1).
        t = length - 1 - i
        offsets = t * batch_size * dim + col_offsets

        grad_y = tl.load(grad_y_ptr + offsets, mask=mask, other=0)
        if t > 0:
            s = tl.load(
                y_ptr + offsets - batch_size * dim, mask=mask, other=0)
        else:
            s = tl.load(s_ptr + col_offsets, mask=mask, other=0)

        grad_s = grad_y + grad_s
        grad_x = grad_s
        grad_Lambda += grad_s * s
        grad_s = grad_s * Lambda

        tl.store(grad_x_ptr + offsets, grad_x, mask=mask)

    tl.store(grad_s_ptr + col_offsets, grad_s, mask=mask)
    tl.store(grad_lambda_ptr + col_offsets, grad_Lambda, mask=mask)

@triton.jit
def diag_ssm_forward_kernel_complex(s_ptr, x_ptr, y_ptr, lambda_ptr,
                                    length, batch_size, dim,
                                    BLOCK_SIZE: tl.constexpr):
    """
    Forward kernel (complex version)

    Arguments:
        s_ptr: [batch_size, dim, 2]
        x_ptr: [length, batch_size, dim, 2]
        lambda_ptr: [dim, 2]
        y_ptr: [length, batch_size, dim, 2]
    """
    # Widen before every shape product, time loop and pointer offset.
    length = tl.cast(length, tl.int64)
    batch_size = tl.cast(batch_size, tl.int64)
    dim = tl.cast(dim, tl.int64)
    col_idx = tl.cast(tl.program_id(0), tl.int64) * BLOCK_SIZE
    col_offsets = col_idx + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < batch_size * dim

    # Load the real and imaginary parts of s and Lambda.
    s_real = tl.load(s_ptr + col_offsets * 2, mask=mask, other=0)
    s_imag = tl.load(s_ptr + col_offsets * 2 + 1, mask=mask, other=0)
    lambda_real = tl.load(
        lambda_ptr + (col_offsets % dim) * 2, mask=mask, other=0)
    lambda_imag = tl.load(
        lambda_ptr + (col_offsets % dim) * 2 + 1, mask=mask, other=0)

    for t in range(length):
        offsets = (t * batch_size * dim + col_offsets) * 2
        # Load the real and imaginary parts of x.
        x_real = tl.load(x_ptr + offsets, mask=mask, other=0)
        x_imag = tl.load(x_ptr + offsets + 1, mask=mask, other=0)

        # Complex multiplication and addition
        new_s_real = s_real * lambda_real - s_imag * lambda_imag + x_real
        new_s_imag = s_real * lambda_imag + s_imag * lambda_real + x_imag

        # Store the updated real and imaginary parts.
        tl.store(y_ptr + offsets, new_s_real, mask=mask)
        tl.store(y_ptr + offsets + 1, new_s_imag, mask=mask)

        # Update s for the next iteration.
        s_real, s_imag = new_s_real, new_s_imag

@triton.jit
def diag_ssm_backward_kernel_complex(
        s_ptr, lambda_ptr, y_ptr, grad_s_ptr, grad_x_ptr, grad_lambda_ptr,
        grad_y_ptr, length, batch_size, dim, BLOCK_SIZE: tl.constexpr):
    """
    Backward kernel (complex version)

    Arguments:
        s_ptr: [batch_size, dim, 2]
        lambda_ptr: [dim, 2]
        y_ptr: [length, batch_size, dim, 2]
        grad_s_ptr: [batch_size, dim, 2]
        grad_x_ptr: [length, batch_size, dim, 2]
        grad_lambda_ptr: [batch_size, dim, 2]
        grad_y_ptr: [length, batch_size, dim, 2]
    """

    # Compute the conjugate Wirtinger derivative \partial f / \partial z^*.
    # Therefore, conjugation is required during the computation.
    # Reference: https://pytorch.org/docs/stable/notes/autograd.html#autograd-for-complex-numbers
    # Negate the imaginary part when loading or storing gradients.

    # Widen before every shape product, time loop and pointer offset.
    length = tl.cast(length, tl.int64)
    batch_size = tl.cast(batch_size, tl.int64)
    dim = tl.cast(dim, tl.int64)
    col_idx = tl.cast(tl.program_id(0), tl.int64) * BLOCK_SIZE
    col_offsets = col_idx + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < batch_size * dim

    # Load the real and imaginary parts of Lambda.
    lambda_real = tl.load(
        lambda_ptr + (col_offsets % dim) * 2, mask=mask, other=0)
    lambda_imag = tl.load(
        lambda_ptr + (col_offsets % dim) * 2 + 1, mask=mask, other=0)

    # Initialize gradients to zero.
    grad_s_real = tl.zeros_like(lambda_real)
    grad_s_imag = tl.zeros_like(lambda_imag)
    grad_lambda_real = tl.zeros_like(lambda_real)
    grad_lambda_imag = tl.zeros_like(lambda_imag)

    for i in range(length):
        # Triton does not support range(length - 1, -1, -1).
        t = length - 1 - i
        offsets = (t * batch_size * dim + col_offsets) * 2

        grad_y_real = tl.load(grad_y_ptr + offsets, mask=mask, other=0)
        grad_y_imag = -tl.load(
            grad_y_ptr + offsets + 1, mask=mask, other=0)
        if t > 0:
            s_real = tl.load(
                y_ptr + offsets - 2 * batch_size * dim, mask=mask, other=0)
            s_imag = tl.load(
                y_ptr + offsets - 2 * batch_size * dim + 1,
                mask=mask,
                other=0)
        else:
            s_real = tl.load(s_ptr + 2 * col_offsets, mask=mask, other=0)
            s_imag = tl.load(
                s_ptr + 2 * col_offsets + 1, mask=mask, other=0)

        grad_s_real = grad_y_real + grad_s_real
        grad_s_imag = grad_y_imag + grad_s_imag
        grad_x_real = grad_s_real
        grad_x_imag = grad_s_imag
        grad_lambda_real += grad_s_real * s_real - grad_s_imag * s_imag
        grad_lambda_imag += grad_s_real * s_imag + grad_s_imag * s_real
        grad_s_real = grad_x_real * lambda_real - grad_x_imag * lambda_imag
        grad_s_imag = grad_x_real * lambda_imag + grad_x_imag * lambda_real

        tl.store(grad_x_ptr + offsets, grad_x_real, mask=mask)
        tl.store(grad_x_ptr + offsets + 1, -grad_x_imag, mask=mask)

    # Store the final gradients.
    tl.store(grad_s_ptr + col_offsets * 2, grad_s_real, mask=mask)
    tl.store(grad_s_ptr + col_offsets * 2 + 1, -grad_s_imag, mask=mask)
    tl.store(
        grad_lambda_ptr + col_offsets * 2, grad_lambda_real, mask=mask)
    tl.store(
        grad_lambda_ptr + col_offsets * 2 + 1,
        -grad_lambda_imag,
        mask=mask)

def _check_tensor(tensor, *, contiguous):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("expected a torch.Tensor")
    if tensor.layout != torch.strided:
        raise ValueError("expected strided layout")
    if tensor.dtype not in (torch.float32, torch.complex64):
        raise ValueError("expected float32 or complex64")
    if not tensor.is_cuda:
        raise ValueError("expected a CUDA tensor")
    if contiguous and not tensor.is_contiguous():
        raise ValueError("expected a contiguous tensor")
    if tensor.is_neg() or tensor.is_conj():
        raise ValueError("unresolved negative/conjugate views are unsupported")
    if tensor.numel() and tensor.data_ptr() % 4:
        raise ValueError("expected a 4-byte aligned data pointer")


def _check_inputs(s, x, Lambda):
    for tensor in (s, x, Lambda):
        _check_tensor(tensor, contiguous=True)
    if not (s.dtype == x.dtype == Lambda.dtype):
        raise ValueError("input dtypes must match")
    if not (s.device == x.device == Lambda.device):
        raise ValueError("input devices must match")
    if x.ndim != 3:
        raise ValueError("x must have shape [length, batch, dim]")
    length, batch_size, dim = x.shape
    if s.shape != (batch_size, dim) or Lambda.shape != (dim,):
        raise ValueError("expected s[batch, dim] and Lambda[dim]")
    return length, batch_size, dim


class _ssm_forward(torch.autograd.Function):
    # TODO: Use @triton.autotune to select the best BLOCK_SIZE.
    # BLOCK_SIZE = 128 appears to work well on an RTX 3090.
    BLOCK_SIZE = 128

    @staticmethod
    def forward(ctx, s, x, Lambda):
        length, batch_size, dim = _check_inputs(s, x, Lambda)
        n = batch_size * dim
        y = torch.zeros_like(x)
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']), )

        with torch.cuda.device(x.device):
            if Lambda.dtype == torch.complex64:
                # Ensure s and x are complex tensors.
                if not torch.is_complex(s):
                    raise ValueError("s must be a complex tensor when Lambda is complex")
                if not torch.is_complex(x):
                    raise ValueError("x must be a complex tensor when Lambda is complex")
                diag_ssm_forward_kernel_complex[grid](
                    torch.view_as_real(s), torch.view_as_real(x),
                    torch.view_as_real(y), torch.view_as_real(Lambda), length,
                    batch_size, dim, _ssm_forward.BLOCK_SIZE)
            elif Lambda.dtype.is_floating_point:
                diag_ssm_forward_kernel[grid](s, x, Lambda, y, length,
                                              batch_size, dim,
                                              _ssm_forward.BLOCK_SIZE)
            else:
                raise ValueError("Unsupported dtype: %s" % Lambda.dtype)
        ctx.save_for_backward(s, y, Lambda)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        s, y, Lambda = ctx.saved_tensors
        length, batch_size, dim = y.shape
        # The original autograd path already materializes a noncontiguous
        # cotangent. No additional input or logical-view copies are introduced.
        _check_tensor(grad_y, contiguous=False)
        if grad_y.dtype != y.dtype or grad_y.device != y.device or grad_y.shape != y.shape:
            raise ValueError("grad_y must match the saved output")
        grad_y = grad_y.contiguous()
        n = batch_size * dim
        grad_s = torch.empty_like(s)
        grad_x = torch.empty_like(grad_y)
        # grad_lambda stores the Lambda gradient for each batch.
        # Sum these values after the kernel finishes.
        grad_lambda = torch.empty_like(s)
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']), )
        with torch.cuda.device(y.device):
            if Lambda.dtype == torch.complex64:
                diag_ssm_backward_kernel_complex[grid](
                    torch.view_as_real(s), torch.view_as_real(Lambda),
                    torch.view_as_real(y), torch.view_as_real(grad_s),
                    torch.view_as_real(grad_x),
                    torch.view_as_real(grad_lambda),
                    torch.view_as_real(grad_y), length, batch_size, dim,
                    _ssm_forward.BLOCK_SIZE)
            else:
                diag_ssm_backward_kernel[grid](
                    s, Lambda, y, grad_s, grad_x, grad_lambda, grad_y, length,
                    batch_size, dim, _ssm_forward.BLOCK_SIZE)
        return grad_s, grad_x, grad_lambda.sum(dim=0)

diag_ssm_forward_triton = _ssm_forward.apply
