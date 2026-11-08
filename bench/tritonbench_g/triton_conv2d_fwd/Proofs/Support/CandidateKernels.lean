import bench.tritonbench_g.triton_conv2d_fwd.Kernels

namespace VeriTile.Bench.Repairs.Conv2d.CandidateHalf
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
  Pasted.candidateKernel dtype Input Weight Output
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

abbrev Parameters := KernelHalf.Parameters

def source (p : Parameters) : ComputeKernel :=
  typed .fp32 p.Input p.Weight p.Output
    p.batch_dim p.in_feat_dim p.in_height p.in_width p.out_feat_dim p.out_height p.out_width p.input_batch_stride p.input_in_feat_stride p.input_height_stride p.input_width_stride p.weight_out_feat_stride p.weight_in_feat_stride p.weight_height_stride p.weight_width_stride p.output_batch_stride p.output_out_feat_stride p.output_height_stride p.output_width_stride p.kernel_height p.kernel_width p.stride_height p.stride_width p.padding_height p.padding_width p.groups Bool.true p.tf32
    p.BLOCK_SIZE_BATCH_HEIGHT_WIDTH p.BLOCK_SIZE_IN_FEAT p.BLOCK_SIZE_OUT_FEAT


end
end VeriTile.Bench.Repairs.Conv2d.CandidateHalf

namespace VeriTile.Bench.Repairs.Conv2d.CandidateDefault
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
  Pasted.candidateKernel dtype Input Weight Output
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

abbrev Parameters := CandidateHalf.Parameters

def source (p : Parameters) : ComputeKernel :=
  typed .fp32 p.Input p.Weight p.Output
    p.batch_dim p.in_feat_dim p.in_height p.in_width p.out_feat_dim p.out_height p.out_width p.input_batch_stride p.input_in_feat_stride p.input_height_stride p.input_width_stride p.weight_out_feat_stride p.weight_in_feat_stride p.weight_height_stride p.weight_width_stride p.output_batch_stride p.output_out_feat_stride p.output_height_stride p.output_width_stride p.kernel_height p.kernel_width p.stride_height p.stride_width p.padding_height p.padding_width p.groups p.tf32
    p.BLOCK_SIZE_BATCH_HEIGHT_WIDTH p.BLOCK_SIZE_IN_FEAT p.BLOCK_SIZE_OUT_FEAT


end
end VeriTile.Bench.Repairs.Conv2d.CandidateDefault

namespace VeriTile.Bench.Repairs.Conv2d.CandidateSource
open VeriTile.Triton
abbrev Parameters := CandidateHalf.Parameters
/-- One selected readable source for both arithmetic interpretations. The
original constexpr FP16 branch determines the operand type before dot. -/
def kernel (fp16 : Bool) (p : Parameters) : ComputeKernel :=
  if fp16 then CandidateHalf.source p else CandidateDefault.source p
end VeriTile.Bench.Repairs.Conv2d.CandidateSource
