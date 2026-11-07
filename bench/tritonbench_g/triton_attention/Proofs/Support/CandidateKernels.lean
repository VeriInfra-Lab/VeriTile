import bench.tritonbench_g.triton_attention.Proofs.Support.KernelEntryAdapters

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.CandidateSource
open VeriTile Triton FP.Structural
noncomputable section

/-- Existing shifted address candidate, on its original 128-row launch. -/
def preprocessKernel (c : Shape) : ComputeKernel :=
  Source.preprocessShiftedTypedKernel .fp32 .fp32 "Out" "DO" "L" "NewDO" "Delta" c.dim
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.CandidateSource
