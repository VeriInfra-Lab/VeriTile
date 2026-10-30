import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_forward_shifted_kernel(
    input_pointer, weight_pointer, output_pointer,
    batch_dim, in_feat_dim, in_height, in_width,
    out_feat_dim, out_height, out_width,
    input_batch_stride, input_in_feat_stride, input_height_stride, input_width_stride,
    weight_out_feat_stride, weight_in_feat_stride, weight_height_stride, weight_width_stride,
    output_batch_stride, output_out_feat_stride, output_height_stride, output_width_stride,
    kernel_height: tl.constexpr, kernel_width: tl.constexpr,
    stride_height: tl.constexpr, stride_width: tl.constexpr,
    padding_height: tl.constexpr, padding_width: tl.constexpr,
    groups: tl.constexpr, fp16: tl.constexpr, tf32: tl.constexpr,
    BLOCK_SIZE_BATCH_HEIGHT_WIDTH: tl.constexpr, BLOCK_SIZE_IN_FEAT: tl.constexpr,
    BLOCK_SIZE_OUT_FEAT: tl.constexpr,
    ):
    """
    2D-convolves over the input using weights.

    Args:
        input_pointer: Pointer to the input to convolve over.
            The input must be of shape [batch_dim, in_feat_dim, in_height, in_width].
        weight_pointer: Pointer to the weights input is convolved over by.
            The weights must be of shape [out_feat_dim, in_feat_dim/groups, kernel_height, kernel_width].
        output_pointer: Pointer to a container the result is written to.
            The container must be of shape [batch_dim, out_feat_dim, out_height, out_width].
        batch_dim: Batch dimension of the input and output.
        in_feat_dim: Dimensionality of the input features.
        in_height: Input height.
        in_width: Input width.
        out_feat_dim: Dimensionality of the output features.
        out_height: Output height.
        out_width: Output width.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_in_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        input_height_stride: Stride necessary to jump one element along the
            input's height dimension.
        input_width_stride: Stride necessary to jump one element along the
            input's width dimension.
        weight_out_feat_stride: Stride necessary to jump one element along the
            weights' output feature dimension.
        weight_in_feat_stride: Stride necessary to jump one element along the
            weights' input feature dimension.
        weight_height_stride: Stride necessary to jump one element along the
            weights' height dimension.
        weight_width_stride: Stride necessary to jump one element along the
            weights' width dimension.
        output_batch_stride: Stride necessary to jump one element along the
            output's batch dimension.
        output_out_feat_stride: Stride necessary to jump one element along the
            output's feature dimension.
        output_height_stride: Stride necessary to jump one element along the
            output's height dimension.
        output_width_stride: Stride necessary to jump one element along the
            output's width dimension.
        kernel_height: Kernel height.
        kernel_width: Kernel width.
        stride_height: Stride of kernel across the height dimension.
        stride_width: Stride of kernel across the width dimension.
        padding_height: Padding applied to the input across the height dimension.
        padding_width: Padding applied to the input across the width dimension.
        groups: Number of groups for the convolution.
        fp16: Flag for loading the input and weights in FP16.
        tf32: Flag for performing matrix products in TF32.
        BLOCK_SIZE_BATCH_HEIGHT_WIDTH: Block size across the batch, height, and
            width dimensions.
        BLOCK_SIZE_IN_FEAT: Block size across the input feature dimension.
        BLOCK_SIZE_OUT_FEAT: Block size across the output feature dimension.
    """
    tl.static_assert(BLOCK_SIZE_BATCH_HEIGHT_WIDTH == 128)
    batch_height_width_pid = tl.program_id(0)
    out_feat_pid = tl.program_id(1)
    group_pid = tl.program_id(2)

    in_group_dim = in_feat_dim // groups
    out_group_dim = out_feat_dim // groups

    batch_height_width_offset = ((batch_height_width_pid << 7) +
                                 tl.arange(0, BLOCK_SIZE_BATCH_HEIGHT_WIDTH))
    batch_height_offset = batch_height_width_offset // out_width
    batch_offset = batch_height_offset // out_height

    output_feat_offset = (out_feat_pid * BLOCK_SIZE_OUT_FEAT +
                          tl.arange(0, BLOCK_SIZE_OUT_FEAT))
    output_height_offset = batch_height_offset % out_height
    output_width_offset = batch_height_width_offset % out_width

    input_pointer += (input_batch_stride * batch_offset +
                      input_in_feat_stride * group_pid * in_group_dim)[:, None]
    weight_pointer += (weight_out_feat_stride * output_feat_offset +
                       weight_out_feat_stride * group_pid * out_group_dim)[None, :]

    accum = tl.zeros((BLOCK_SIZE_BATCH_HEIGHT_WIDTH, BLOCK_SIZE_OUT_FEAT),
                     dtype=tl.float32)

    for h in range(kernel_height):
        for w in range(kernel_width):
            for c in range(0, in_group_dim, BLOCK_SIZE_IN_FEAT):
                input_feat_offset = c + tl.arange(0, BLOCK_SIZE_IN_FEAT)
                input_height_offset = (h - padding_height +
                                       stride_height * output_height_offset)
                input_width_offset = (w - padding_width +
                                      stride_width * output_width_offset)

                curr_input_pointer = (input_pointer +
                                     (input_in_feat_stride * input_feat_offset)[None, :] +
                                     (input_height_stride * input_height_offset)[:, None] +
                                     (input_width_stride * input_width_offset)[:, None])
                curr_weight_pointer = (weight_pointer +
                                      (weight_in_feat_stride * input_feat_offset)[:, None] +
                                      (weight_height_stride * h) +
                                      (weight_width_stride * w))

                input_mask = ((batch_offset < batch_dim)[:, None] &
                              (input_feat_offset < in_group_dim)[None, :] &
                              (0 <= input_height_offset)[:, None] &
                              (input_height_offset < in_height)[:, None] &
                              (0 <= input_width_offset)[:, None] &
                              (input_width_offset < in_width)[:, None])
                weight_mask = ((input_feat_offset < in_group_dim)[:, None] &
                               (output_feat_offset < out_group_dim)[None, :])

                input_block = tl.load(curr_input_pointer, mask=input_mask, other=0.0)
                weight_block = tl.load(curr_weight_pointer, mask=weight_mask, other=0.0)

                if fp16:
                    input_block = input_block.to(tl.float16)
                    weight_block = weight_block.to(tl.float16)

                accum += tl.dot(input_block, weight_block, allow_tf32=tf32)

    output_pointer += ((output_batch_stride * batch_offset)[:, None] +
                       (output_out_feat_stride * (group_pid * out_group_dim + output_feat_offset))[None, :] +
                       (output_height_stride * output_height_offset)[:, None] +
                       (output_width_stride * output_width_offset)[:, None])
    output_mask = ((batch_offset < batch_dim)[:, None] &
                   (output_feat_offset < out_group_dim)[None, :] &
                   (output_height_offset < out_height)[:, None] &
                   (output_width_offset < out_width)[:, None])

    tl.store(output_pointer, accum, mask=output_mask)


def _tensor_contract(tensor):
    if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
            or tensor.ndim != 4 or not tensor.is_cuda
            or tensor.dtype != torch.float32 or tensor.is_neg() or tensor.is_conj()):
        raise ValueError("resolved rank-four FP32 CUDA tensors required")
    if any(s <= 0 for s in tensor.stride()):
        raise ValueError("positive tensor strides required")
    storage = tensor.untyped_storage()
    base, capacity, pointer = storage.data_ptr(), storage.nbytes(), tensor.data_ptr()
    if tensor.numel():
        top = sum((d - 1) * s for d, s in zip(tensor.shape, tensor.stride()))
        if (pointer % 4 or pointer < base or pointer + 4 * (top + 1) > base + capacity
                or base + capacity > 2**64):
            raise ValueError("accessed FP32 elements must lie in actual backing storage")


def _validate_inputs(input_tensor, weight_tensor, kernel_height, kernel_width,
                     stride_height, stride_width, padding_height, padding_width,
                     groups, fp16, tf32):
    _tensor_contract(input_tensor)
    _tensor_contract(weight_tensor)
    if input_tensor.device != weight_tensor.device:
        raise ValueError("input and weight must share a CUDA device")
    if input_tensor.device.index != torch.cuda.current_device():
        raise ValueError("input device must be the current CUDA device")
    if type(fp16) is not bool or type(tf32) is not bool:
        raise TypeError("fp16 and tf32 must be Boolean flags")
    positives = (kernel_height, kernel_width, stride_height, stride_width, groups)
    if any(type(v) is not int or v <= 0 for v in positives):
        raise ValueError("kernel, stride, and groups must be positive integers")
    if any(type(v) is not int or v < 0 for v in (padding_height, padding_width)):
        raise ValueError("padding must be nonnegative integers")
    batch, channels, height, width = input_tensor.shape
    outputs = weight_tensor.shape[0]
    if min(channels, height, width, outputs) <= 0 or channels % groups or outputs % groups:
        raise ValueError("positive channels/spatial extents and divisible groups required")
    cg, og = channels // groups, outputs // groups
    if weight_tensor.shape != (outputs, cg, kernel_height, kernel_width):
        raise ValueError("weight must have shape [O,C/groups,KH,KW]")
    oh = (height + 2 * padding_height - kernel_height) // stride_height + 1
    ow = (width + 2 * padding_width - kernel_width) // stride_width + 1
    if oh <= 0 or ow <= 0:
        raise ValueError("output spatial dimensions must be positive")
    if batch and (triton.cdiv(outputs, 32) > 65535 or groups > 65535):
        raise ValueError("direct CUDA grid Y and Z dimensions must not exceed 65535")
    sn, sc, sh, sw = input_tensor.stride()
    to, tc, th, tw = weight_tensor.stride()
    padded_spatial = 128 * triton.cdiv(batch * oh * ow, 128)
    padded_features = 32 * triton.cdiv(outputs, 32)
    padded_channels = 32 * triton.cdiv(cg, 32)
    padded_batch = padded_spatial // (oh * ow) + 1
    height_abs = kernel_height + padding_height + stride_height * oh
    width_abs = kernel_width + padding_width + stride_width * ow
    envelope = (batch + channels + height + width + outputs + oh + ow
                + kernel_height + kernel_width + stride_height + stride_width
                + padding_height + padding_width + groups + cg + og + 128
                + height + 2 * padding_height + width + 2 * padding_width
                + padded_spatial + padded_features + padded_channels + padded_batch
                + height_abs + width_abs + sn + sc + sh + sw + to + tc + th + tw
                + padded_batch * sn + groups * cg * sc + padded_channels * sc
                + height_abs * sh + width_abs * sw
                + (padded_features + groups * og) * to + padded_channels * tc
                + kernel_height * th + kernel_width * tw
                + padded_batch * outputs * oh * ow
                + (groups * og + padded_features) * oh * ow + oh * ow + ow)
    if envelope >= 2**31:
        raise ValueError("signed32 envelope exceeded for formed padded source expressions")

def conv2d_forward_shifted(input_tensor, weight_tensor, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width, groups, fp16=False, tf32=False):
    _validate_inputs(input_tensor, weight_tensor, kernel_height, kernel_width, stride_height, stride_width, padding_height, padding_width, groups, fp16, tf32)
    # Extract dimensions
    batch_dim, in_feat_dim, in_height, in_width = input_tensor.shape
    out_feat_dim, _, _, _ = weight_tensor.shape
    out_height = (in_height + 2 * padding_height - kernel_height) // stride_height + 1
    out_width = (in_width + 2 * padding_width - kernel_width) // stride_width + 1

    # Calculate strides
    input_batch_stride, input_in_feat_stride, input_height_stride, input_width_stride = input_tensor.stride()
    weight_out_feat_stride, weight_in_feat_stride, weight_height_stride, weight_width_stride = weight_tensor.stride()
    output_tensor = torch.empty((batch_dim, out_feat_dim, out_height, out_width), device=input_tensor.device, dtype=torch.float32)
    output_batch_stride, output_out_feat_stride, output_height_stride, output_width_stride = output_tensor.stride()
    if batch_dim == 0:
        return output_tensor

    # Define block sizes
    BLOCK_SIZE_BATCH_HEIGHT_WIDTH = 128
    BLOCK_SIZE_IN_FEAT = 32
    BLOCK_SIZE_OUT_FEAT = 32

    # Launch the Triton kernel
    grid = (triton.cdiv(batch_dim * out_height * out_width, BLOCK_SIZE_BATCH_HEIGHT_WIDTH),
            triton.cdiv(out_feat_dim, BLOCK_SIZE_OUT_FEAT),
            groups)
    
    conv2d_forward_shifted_kernel[grid](
        input_tensor, weight_tensor, output_tensor,
        batch_dim, in_feat_dim, in_height, in_width,
        out_feat_dim, out_height, out_width,
        input_batch_stride, input_in_feat_stride, input_height_stride, input_width_stride,
        weight_out_feat_stride, weight_in_feat_stride, weight_height_stride, weight_width_stride,
        output_batch_stride, output_out_feat_stride, output_height_stride, output_width_stride,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_height, padding_width,
        groups, fp16, tf32,
        BLOCK_SIZE_BATCH_HEIGHT_WIDTH, BLOCK_SIZE_IN_FEAT, BLOCK_SIZE_OUT_FEAT
    )
    
    return output_tensor
