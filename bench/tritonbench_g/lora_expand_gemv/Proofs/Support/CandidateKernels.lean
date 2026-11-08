import bench.tritonbench_g.lora_expand_gemv.Kernels
namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LoraExpandGemv
noncomputable section
/-- Existing Python specialization: SPLIT_N=N and the checked caller has N>0. -/
def unitSplitSource (p : Branches.Parameters) : ComputeKernel :=
  Pasted.candidateKernel .single .half .single .i64 p.input p.weight p.output p.indices.cast
    p.outputSize p.inner p.inputBatchStride p.inputInnerStride p.loraStride
    p.weightRowStride p.weightInnerStride p.outputBatchStride p.outputStride
    p.blockRows p.blockInner p.splits p.evenK p.addInputs p.castType
end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated
