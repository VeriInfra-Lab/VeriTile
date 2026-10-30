import torch
import triton
import triton.language as tl

@triton.jit
def fused_add_mul_activation_kernel(x_ptr, bias_ptr, in_ptr,
                                    num_weights: tl.constexpr,
                                    xnumel: tl.constexpr,
                                    multiplier: tl.constexpr,
                                    activation: tl.constexpr,
                                    BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    index = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    mask = index < xnumel
    bias_index = index % num_weights
    tmp0 = tl.load(x_ptr + index, mask)
    tmp1 = tl.load(bias_ptr + bias_index, mask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr + index, mask)
    activ_input = multiplier * tmp3 + tmp0 + tmp1
    if activation == "sigmoid":
        ma_result = tl.sigmoid(activ_input)
        # option 2 - calculate sigmoid using exp
        # ma_result = 1.0 / (1.0 + tl.exp(-sigmoid_input))
        # option 3: fast sigmoid - inaccurate but faster
        # ma_result = 1.0 / (1.0 + tl.abs(sigmoid_input))
    elif activation == "relu":
        ma_result = tl.maximum(0, activ_input)

    tl.store(x_ptr + index, ma_result, mask)



def fused_add_mul_activation_torch(in_out_tensor: torch.Tensor, bias: torch.Tensor,
                                   in_tensor: torch.Tensor) -> torch.Tensor:
    """Update and return the original contiguous FP32 tensor with cyclic bias."""
    tensors = (in_out_tensor, bias, in_tensor)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("fused activation requires Tensor inputs")
    if any(t.layout != torch.strided for t in tensors):
        raise ValueError("fused activation requires strided tensors")
    if any(t.dtype != torch.float32 for t in tensors):
        raise TypeError("fused activation requires FP32 tensors")
    if not all(t.is_cuda for t in tensors):
        raise ValueError("fused activation requires CUDA tensors")
    if in_out_tensor.device != bias.device or in_out_tensor.device != in_tensor.device:
        raise ValueError("fused activation requires one CUDA device")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("fused activation requires contiguous tensors")
    if any(t.is_neg() for t in tensors):
        raise ValueError("fused activation does not accept negative views")
    n_elements = in_out_tensor.numel()
    num_weights = bias.numel()
    if in_tensor.numel() < n_elements:
        raise ValueError("in_tensor must cover the x prefix")
    if n_elements >= 2**31 or num_weights >= 2**31:
        raise ValueError("x and bias must have fewer than 2**31 elements")
    if n_elements == 0:
        return in_out_tensor
    if num_weights == 0:
        raise ValueError("nonempty x requires a nonempty bias")
    xptr, bptr, iptr = (t.data_ptr() for t in tensors)
    if xptr % 4 or bptr % 4 or iptr % 4:
        raise ValueError("fused activation requires 4-byte-aligned pointers")
    xend = xptr + 4*n_elements
    if not ((xend <= bptr or bptr+4*num_weights <= xptr) and
            (xend <= iptr or iptr+4*n_elements <= xptr)):
        raise ValueError("x must be disjoint from bias and the input prefix")
    BLOCK_SIZE = min(2048, triton.next_power_of_2(n_elements))
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    with torch.cuda.device(in_out_tensor.device):
        fused_add_mul_activation_kernel[grid](in_out_tensor, bias, in_tensor,
                                              num_weights, n_elements,
                                              multiplier=0.5,
                                              activation="sigmoid",
                                              BLOCK_SIZE=BLOCK_SIZE,
                                              enable_fp_fusion=True)
    return in_out_tensor


@triton.jit
def fused_add_mul_activation_commuted_kernel(x_ptr, bias_ptr, in_ptr,
                                    num_weights: tl.constexpr,
                                    xnumel: tl.constexpr,
                                    multiplier: tl.constexpr,
                                    activation: tl.constexpr,
                                    BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    index = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    mask = index < xnumel
    bias_index = index % num_weights
    tmp0 = tl.load(x_ptr + index, mask)
    tmp1 = tl.load(bias_ptr + bias_index, mask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr + index, mask)
    activ_input = tmp1 + (multiplier * tmp3 + tmp0)
    if activation == "sigmoid":
        ma_result = tl.sigmoid(activ_input)
        # option 2 - calculate sigmoid using exp
        # ma_result = 1.0 / (1.0 + tl.exp(-sigmoid_input))
        # option 3: fast sigmoid - inaccurate but faster
        # ma_result = 1.0 / (1.0 + tl.abs(sigmoid_input))
    elif activation == "relu":
        ma_result = tl.maximum(0, activ_input)

    tl.store(x_ptr + index, ma_result, mask)
