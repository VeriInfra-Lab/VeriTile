import bench.repairs.tritonbench_g.triton_conv2d_fwd.Kernels
import VeriTile.Triton.DSL.CallSignature

namespace VeriTile.Bench.Repairs.Conv2d.RawPrototype
open VeriTile.Triton
noncomputable section
def signature : CallSignature where
  pointers := [("input_pointer",.inputFloat),("weight_pointer",.inputFloat),("output_pointer",.inputFloat)]
  scalars := [("batch_dim",.integer),("in_feat_dim",.integer),("in_height",.integer),("in_width",.integer),("out_feat_dim",.integer),("out_height",.integer),("out_width",.integer),("input_batch_stride",.integer),("input_in_feat_stride",.integer),("input_height_stride",.integer),("input_width_stride",.integer),("weight_out_feat_stride",.integer),("weight_in_feat_stride",.integer),("weight_height_stride",.integer),("weight_width_stride",.integer),("output_batch_stride",.integer),("output_out_feat_stride",.integer),("output_height_stride",.integer),("output_width_stride",.integer)]
  constexpr := ["kernel_height","kernel_width","stride_height","stride_width","padding_height","padding_width","groups","fp16","tf32","BLOCK_SIZE_BATCH_HEIGHT_WIDTH","BLOCK_SIZE_IN_FEAT","BLOCK_SIZE_OUT_FEAT"]
  specializeBooleans := ["fp16"]
  readPorts := ["input_pointer","weight_pointer"]
  writePorts := ["output_pointer"]

def half (dtype : FloatDType) (p : KernelHalf.Parameters) : ComputeKernel :=
  let input_pointer : RegionName := p.Input
  let weight_pointer : RegionName := p.Weight
  let output_pointer : RegionName := p.Output
  let kernel_height : Nat := p.kernel_height
  let kernel_width : Nat := p.kernel_width
  let stride_height : Nat := p.stride_height
  let stride_width : Nat := p.stride_width
  let padding_height : Int := (p.padding_height : Int)
  let padding_width : Int := (p.padding_width : Int)
  let groups : Nat := p.groups
  let fp16 : Bool := Bool.true
  let tf32 : Bool := p.tf32
  let BLOCK_SIZE_BATCH_HEIGHT_WIDTH : Nat := p.BLOCK_SIZE_BATCH_HEIGHT_WIDTH
  let BLOCK_SIZE_IN_FEAT : Nat := p.BLOCK_SIZE_IN_FEAT
  let BLOCK_SIZE_OUT_FEAT : Nat := p.BLOCK_SIZE_OUT_FEAT
  with_triton_call signature call_args [batch_dim := p.batch_dim,in_feat_dim := p.in_feat_dim,in_height := p.in_height,in_width := p.in_width,out_feat_dim := p.out_feat_dim,out_height := p.out_height,out_width := p.out_width,input_batch_stride := p.input_batch_stride,input_in_feat_stride := p.input_in_feat_stride,input_height_stride := p.input_height_stride,input_width_stride := p.input_width_stride,weight_out_feat_stride := p.weight_out_feat_stride,weight_in_feat_stride := p.weight_in_feat_stride,weight_height_stride := p.weight_height_stride,weight_width_stride := p.weight_width_stride,output_batch_stride := p.output_batch_stride,output_out_feat_stride := p.output_out_feat_stride,output_height_stride := p.output_height_stride,output_width_stride := p.output_width_stride] in triton {
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
    batch_height_width_pid = tl.program_id(0)
    out_feat_pid = tl.program_id(1)
    group_pid = tl.program_id(2)

    in_group_dim = in_feat_dim // groups
    out_group_dim = out_feat_dim // groups

    batch_height_width_offset = (batch_height_width_pid * BLOCK_SIZE_BATCH_HEIGHT_WIDTH +
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
  }
end
end VeriTile.Bench.Repairs.Conv2d.RawPrototype
