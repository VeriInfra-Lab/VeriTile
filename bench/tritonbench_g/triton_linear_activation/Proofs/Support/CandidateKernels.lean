import bench.tritonbench_g.triton_linear_activation.Proofs.Support.SourceKernels

namespace VeriTile.Bench.Repairs.TritonLinearActivation.CandidateSource
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
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
  candidateKernel dtype C ACT_INPUTS A B biasR
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
end VeriTile.Bench.Repairs.TritonLinearActivation.CandidateSource
