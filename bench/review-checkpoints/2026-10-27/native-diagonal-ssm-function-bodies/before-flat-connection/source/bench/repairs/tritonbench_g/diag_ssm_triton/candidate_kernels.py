"""Existing PID/block index commutation candidates, retaining full source bodies."""
import triton
import triton.language as tl

@triton.jit
def diag_ssm_forward_kernel_index_commuted(s_ptr, x_ptr, lambda_ptr, y_ptr, length,
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
    col_idx = BLOCK_SIZE * tl.cast(tl.program_id(0), tl.int64)
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
def diag_ssm_backward_kernel_index_commuted(
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
    col_idx = BLOCK_SIZE * tl.cast(tl.program_id(0), tl.int64)
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
def diag_ssm_forward_kernel_complex_index_commuted(s_ptr, x_ptr, y_ptr, lambda_ptr,
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
    col_idx = BLOCK_SIZE * tl.cast(tl.program_id(0), tl.int64)
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
def diag_ssm_backward_kernel_complex_index_commuted(
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
    col_idx = BLOCK_SIZE * tl.cast(tl.program_id(0), tl.int64)
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

