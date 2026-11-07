import bench.tritonbench_g.context_attn_llama.Proofs.Sources.Mathematical_Program
import bench.tritonbench_g.context_attn_llama.Proofs.Sources.Mathematical_HostInterface

/-! Complete selected FP32 source in the accepted bounded machine-index
projection. Its signed-width transport is unchanged. Other storage cases retain
the existing shared source as compatibility views. -/
namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
open VeriTile.Triton
abbrev Storage := VeriTile.Bench.TritonBenchG.ContextAttnLlama.Storage
abbrev Layout := VeriTile.Bench.TritonBenchG.ContextAttnLlama.Formula.Layout
abbrev Regions := VeriTile.Bench.TritonBenchG.ContextAttnLlama.Program.Regions

/-- Direct binding to the shared selected source. -/
def originalFP32 (p : Layout) (r : Regions) (scale : ℝ) : ComputeKernel :=
  VeriTile.Bench.TritonBenchG.ContextAttnLlama.Program.kernel p r .fp32 Bool.false scale


def blockEndFactoredFP32 (p : Layout) (r : Regions) (scale : ℝ) : ComputeKernel :=
  blockEndFactoredKernel .fp32 r.query r.key r.value scale r.out r.start r.sequence r.pages r.request r.prompt
    p.qRow p.qHead p.qDim p.kRow p.kHead p.kDim p.vRow p.vHead p.vDim
    p.outRow p.outHead p.outDim p.reqRow p.reqToken p.kvGroup p.heads p.D p.BM p.BN


noncomputable def kernel (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) : ComputeKernel :=
  match storage with
  | .fp32 => originalFP32 p r scale
  | .fp16 => VeriTile.Bench.TritonBenchG.ContextAttnLlama.Program.kernel p r .fp16 Bool.false scale
  | .bf16 => VeriTile.Bench.TritonBenchG.ContextAttnLlama.Program.kernel p r .bf16 Bool.false scale
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
