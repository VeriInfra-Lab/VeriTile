import VeriTile.Triton
import bench.tritonbench_g.triton_linear_activation.Kernels
import VeriTile.Triton.Float.Structural
import bench.tritonbench_g.triton_linear_activation.Proofs.Config

namespace VeriTile.Bench.Repairs.TritonLinearActivation.KernelSource
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000

/-- Concrete arguments after successful ordinary preparation. The actual
kernel pointer formats are FP32 normally and FP16 under the explicit AMP cast.
The caller's FP32 saved tensor and any AMP copyback are outside this kernel. -/
structure Parameters where
  (output saved input weight bias : RegionName)
  (rows columns inner : Nat)
  (outputRowStride outputColStride savedRowStride savedColStride : Nat)
  (inputRowStride inputInnerStride weightColStride weightInnerStride : Nat)
  (blockRows groupRows blockColumns blockInner : Nat)
  (hasBias saveInputs : Bool)
  activation : Option String

def dtype : Mode → FloatDType | .fp32 => .fp32 | .autocastFP16 => .fp16

/-- Thin invocation adapter for the complete copied Python body. Cache keys
and SPLIT_K retain the repaired launch values; helper bodies and ordinary stores
remain in Sources/Body.lean. Module sqrt tags keep the existing separate Real
and native constant interpretations. -/
def typed (dtype : FloatDType)
    (C ACT_INPUTS A B biasR : RegionName)
    (M N K output_m_stride output_n_stride act_inputs_m_stride act_inputs_n_stride
      a_m_stride a_k_stride b_n_stride b_k_stride BLOCK_M GROUP_M BLOCK_N BLOCK_K : Nat)
    (HAS_BIAS SHOULD_SAVE_ACT_INPUTS K_LOAD_MASK_NEEDED : Bool) (ACTIVATION : Option String) :
    ComputeKernel :=
  originalKernel dtype C ACT_INPUTS A B biasR
    M N K (M/32) (N/32) (K/32)
    output_m_stride output_n_stride act_inputs_m_stride act_inputs_n_stride
    a_m_stride a_k_stride b_n_stride b_k_stride BLOCK_M GROUP_M BLOCK_N BLOCK_K 1
    HAS_BIAS SHOULD_SAVE_ACT_INPUTS K_LOAD_MASK_NEEDED ACTIVATION

def source (mode : Mode) (p : Parameters) : ComputeKernel :=
  typed (dtype mode) p.output p.saved p.input p.weight p.bias
    p.rows p.columns p.inner p.outputRowStride p.outputColStride p.savedRowStride p.savedColStride
    p.inputRowStride p.inputInnerStride p.weightColStride p.weightInnerStride
    p.blockRows p.groupRows p.blockColumns p.blockInner p.hasBias p.saveInputs
    (p.inner%p.blockInner==0) p.activation


end
end VeriTile.Bench.Repairs.TritonLinearActivation.KernelSource
