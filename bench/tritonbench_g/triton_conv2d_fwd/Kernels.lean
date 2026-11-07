import bench.tritonbench_g.triton_conv2d_fwd.Proofs.Support.Model
import VeriTile.Triton.DSL.StaticAssertions
namespace VeriTile.Bench.Repairs.Conv2d.Pasted
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 40000
def signature (dtype : FloatDType) : CallSignature where
  floatingPointers := [("input_pointer",dtype),("weight_pointer",dtype),("output_pointer",dtype)]
  scalars := [("batch_dim",.integer),("in_feat_dim",.integer),("in_height",.integer),("in_width",.integer),("out_feat_dim",.integer),("out_height",.integer),("out_width",.integer),("input_batch_stride",.integer),("input_in_feat_stride",.integer),("input_height_stride",.integer),("input_width_stride",.integer),("weight_out_feat_stride",.integer),("weight_in_feat_stride",.integer),("weight_height_stride",.integer),("weight_width_stride",.integer),("output_batch_stride",.integer),("output_out_feat_stride",.integer),("output_height_stride",.integer),("output_width_stride",.integer)]
  constexpr := ["kernel_height","kernel_width","stride_height","stride_width","padding_height","padding_width","groups","fp16","tf32","BLOCK_SIZE_BATCH_HEIGHT_WIDTH","BLOCK_SIZE_IN_FEAT","BLOCK_SIZE_OUT_FEAT"]
  readPorts := ["input_pointer","weight_pointer"]
  writePorts := ["output_pointer"]
def originalKernel (dtype : FloatDType) (input_pointer weight_pointer output_pointer : RegionName)
    (batch_dim in_feat_dim in_height in_width out_feat_dim out_height out_width input_batch_stride input_in_feat_stride input_height_stride input_width_stride weight_out_feat_stride weight_in_feat_stride weight_height_stride weight_width_stride output_batch_stride output_out_feat_stride output_height_stride output_width_stride kernel_height kernel_width stride_height stride_width groups : Nat) (padding_height padding_width : Int) (BLOCK_SIZE_BATCH_HEIGHT_WIDTH BLOCK_SIZE_IN_FEAT BLOCK_SIZE_OUT_FEAT : Nat) (fp16 tf32 : Bool) : ComputeKernel :=
  with_triton_call (signature dtype) call_args [batch_dim := batch_dim,in_feat_dim := in_feat_dim,in_height := in_height,in_width := in_width,out_feat_dim := out_feat_dim,out_height := out_height,out_width := out_width,input_batch_stride := input_batch_stride,input_in_feat_stride := input_in_feat_stride,input_height_stride := input_height_stride,input_width_stride := input_width_stride,weight_out_feat_stride := weight_out_feat_stride,weight_in_feat_stride := weight_in_feat_stride,weight_height_stride := weight_height_stride,weight_width_stride := weight_width_stride,output_batch_stride := output_batch_stride,output_out_feat_stride := output_out_feat_stride,output_height_stride := output_height_stride,output_width_stride := output_width_stride] in triton {
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

def candidateKernel (dtype : FloatDType) (input_pointer weight_pointer output_pointer : RegionName)
    (batch_dim in_feat_dim in_height in_width out_feat_dim out_height out_width input_batch_stride input_in_feat_stride input_height_stride input_width_stride weight_out_feat_stride weight_in_feat_stride weight_height_stride weight_width_stride output_batch_stride output_out_feat_stride output_height_stride output_width_stride kernel_height kernel_width stride_height stride_width groups : Nat) (padding_height padding_width : Int) (BLOCK_SIZE_BATCH_HEIGHT_WIDTH BLOCK_SIZE_IN_FEAT BLOCK_SIZE_OUT_FEAT : Nat) (fp16 tf32 : Bool) : ComputeKernel :=
  with_triton_call (signature dtype) call_args [batch_dim := batch_dim,in_feat_dim := in_feat_dim,in_height := in_height,in_width := in_width,out_feat_dim := out_feat_dim,out_height := out_height,out_width := out_width,input_batch_stride := input_batch_stride,input_in_feat_stride := input_in_feat_stride,input_height_stride := input_height_stride,input_width_stride := input_width_stride,weight_out_feat_stride := weight_out_feat_stride,weight_in_feat_stride := weight_in_feat_stride,weight_height_stride := weight_height_stride,weight_width_stride := weight_width_stride,output_batch_stride := output_batch_stride,output_out_feat_stride := output_out_feat_stride,output_height_stride := output_height_stride,output_width_stride := output_width_stride] in triton {
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
  }
end
end VeriTile.Bench.Repairs.Conv2d.Pasted

namespace VeriTile.Bench.Repairs.Conv2d.KernelHalf
open VeriTile.Triton FP.Structural
noncomputable section
/-- Readable source quotation for the FP16 constexpr specialization.
The repaired masked loads explicitly supply zero; both TF32 flag values and
all three source loops are preserved. Real and native proofs share this source. -/
def typed (dtype : FloatDType)
    (Input Weight Output : RegionName)
    (batch_dim in_feat_dim in_height in_width out_feat_dim out_height out_width
      input_batch_stride input_in_feat_stride input_height_stride input_width_stride
      weight_out_feat_stride weight_in_feat_stride weight_height_stride weight_width_stride
      output_batch_stride output_out_feat_stride output_height_stride output_width_stride
      kernel_height kernel_width stride_height stride_width padding_height padding_width groups : Nat)
    (_fp16 _tf32 : Bool)
    (BLOCK_SIZE_BATCH_HEIGHT_WIDTH BLOCK_SIZE_IN_FEAT BLOCK_SIZE_OUT_FEAT : Nat) :
    ComputeKernel :=
  Pasted.originalKernel dtype Input Weight Output
    (batch_dim := batch_dim) (in_feat_dim := in_feat_dim) (in_height := in_height)
    (in_width := in_width) (out_feat_dim := out_feat_dim) (out_height := out_height)
    (out_width := out_width) (input_batch_stride := input_batch_stride) (input_in_feat_stride := input_in_feat_stride)
    (input_height_stride := input_height_stride) (input_width_stride := input_width_stride) (weight_out_feat_stride := weight_out_feat_stride)
    (weight_in_feat_stride := weight_in_feat_stride) (weight_height_stride := weight_height_stride) (weight_width_stride := weight_width_stride)
    (output_batch_stride := output_batch_stride) (output_out_feat_stride := output_out_feat_stride) (output_height_stride := output_height_stride)
    (output_width_stride := output_width_stride) (kernel_height := kernel_height) (kernel_width := kernel_width)
    (stride_height := stride_height) (stride_width := stride_width) (padding_height := (padding_height : Int))
    (padding_width := (padding_width : Int)) (groups := groups) (BLOCK_SIZE_BATCH_HEIGHT_WIDTH := BLOCK_SIZE_BATCH_HEIGHT_WIDTH)
    (BLOCK_SIZE_IN_FEAT := BLOCK_SIZE_IN_FEAT) (BLOCK_SIZE_OUT_FEAT := BLOCK_SIZE_OUT_FEAT)
    (fp16 := _fp16) (tf32 := _tf32)

structure Parameters where
  (Input Weight Output : RegionName)
  (batch_dim in_feat_dim in_height in_width out_feat_dim out_height out_width input_batch_stride input_in_feat_stride input_height_stride input_width_stride weight_out_feat_stride weight_in_feat_stride weight_height_stride weight_width_stride output_batch_stride output_out_feat_stride output_height_stride output_width_stride kernel_height kernel_width stride_height stride_width padding_height padding_width groups BLOCK_SIZE_BATCH_HEIGHT_WIDTH BLOCK_SIZE_IN_FEAT BLOCK_SIZE_OUT_FEAT : Nat)
  tf32 : Bool

def source (p : Parameters) : ComputeKernel :=
  typed .fp32 p.Input p.Weight p.Output
    p.batch_dim p.in_feat_dim p.in_height p.in_width p.out_feat_dim p.out_height p.out_width p.input_batch_stride p.input_in_feat_stride p.input_height_stride p.input_width_stride p.weight_out_feat_stride p.weight_in_feat_stride p.weight_height_stride p.weight_width_stride p.output_batch_stride p.output_out_feat_stride p.output_height_stride p.output_width_stride p.kernel_height p.kernel_width p.stride_height p.stride_width p.padding_height p.padding_width p.groups Bool.true p.tf32
    p.BLOCK_SIZE_BATCH_HEIGHT_WIDTH p.BLOCK_SIZE_IN_FEAT p.BLOCK_SIZE_OUT_FEAT


end
end VeriTile.Bench.Repairs.Conv2d.KernelHalf

namespace VeriTile.Bench.Repairs.Conv2d.KernelDefault
open VeriTile.Triton FP.Structural
noncomputable section
/-- The same Python source specialized at fp16=false, with FP32 dot operands.
Masked loads explicitly supply zero; both TF32 values and every source loop
are preserved. This specialization introduces no half conversion. -/
def typed (dtype : FloatDType)
    (Input Weight Output : RegionName)
    (batch_dim in_feat_dim in_height in_width out_feat_dim out_height out_width
      input_batch_stride input_in_feat_stride input_height_stride input_width_stride
      weight_out_feat_stride weight_in_feat_stride weight_height_stride weight_width_stride
      output_batch_stride output_out_feat_stride output_height_stride output_width_stride
      kernel_height kernel_width stride_height stride_width padding_height padding_width groups : Nat)
    (_tf32 : Bool)
    (BLOCK_SIZE_BATCH_HEIGHT_WIDTH BLOCK_SIZE_IN_FEAT BLOCK_SIZE_OUT_FEAT : Nat) :
    ComputeKernel :=
  Pasted.originalKernel dtype Input Weight Output
    (batch_dim := batch_dim) (in_feat_dim := in_feat_dim) (in_height := in_height)
    (in_width := in_width) (out_feat_dim := out_feat_dim) (out_height := out_height)
    (out_width := out_width) (input_batch_stride := input_batch_stride) (input_in_feat_stride := input_in_feat_stride)
    (input_height_stride := input_height_stride) (input_width_stride := input_width_stride) (weight_out_feat_stride := weight_out_feat_stride)
    (weight_in_feat_stride := weight_in_feat_stride) (weight_height_stride := weight_height_stride) (weight_width_stride := weight_width_stride)
    (output_batch_stride := output_batch_stride) (output_out_feat_stride := output_out_feat_stride) (output_height_stride := output_height_stride)
    (output_width_stride := output_width_stride) (kernel_height := kernel_height) (kernel_width := kernel_width)
    (stride_height := stride_height) (stride_width := stride_width) (padding_height := (padding_height : Int))
    (padding_width := (padding_width : Int)) (groups := groups) (BLOCK_SIZE_BATCH_HEIGHT_WIDTH := BLOCK_SIZE_BATCH_HEIGHT_WIDTH)
    (BLOCK_SIZE_IN_FEAT := BLOCK_SIZE_IN_FEAT) (BLOCK_SIZE_OUT_FEAT := BLOCK_SIZE_OUT_FEAT)
    (fp16 := Bool.false) (tf32 := _tf32)

abbrev Parameters := KernelHalf.Parameters

def source (p : Parameters) : ComputeKernel :=
  typed .fp32 p.Input p.Weight p.Output
    p.batch_dim p.in_feat_dim p.in_height p.in_width p.out_feat_dim p.out_height p.out_width p.input_batch_stride p.input_in_feat_stride p.input_height_stride p.input_width_stride p.weight_out_feat_stride p.weight_in_feat_stride p.weight_height_stride p.weight_width_stride p.output_batch_stride p.output_out_feat_stride p.output_height_stride p.output_width_stride p.kernel_height p.kernel_width p.stride_height p.stride_width p.padding_height p.padding_width p.groups p.tf32
    p.BLOCK_SIZE_BATCH_HEIGHT_WIDTH p.BLOCK_SIZE_IN_FEAT p.BLOCK_SIZE_OUT_FEAT


end
end VeriTile.Bench.Repairs.Conv2d.KernelDefault

namespace VeriTile.Bench.Repairs.Conv2d.KernelSource
open VeriTile.Triton
abbrev Parameters := KernelHalf.Parameters
/-- One selected readable source for both arithmetic interpretations. The
original constexpr FP16 branch determines the operand type before dot. -/
def kernel (fp16 : Bool) (p : Parameters) : ComputeKernel :=
  if fp16 then KernelHalf.source p else KernelDefault.source p
end VeriTile.Bench.Repairs.Conv2d.KernelSource
