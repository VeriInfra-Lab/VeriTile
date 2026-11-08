import bench.tritonbench_g.context_attn_fwd.Proofs.Sources.Mathematical_Program

/-! Complete selected FP32 source in the existing bounded machine-index projection.
The accepted signed-width transport remains unchanged. Other storage branches
are compatibility views of the existing shared source. -/
namespace VeriTile.Bench.Repairs.ContextAttnFwd
open VeriTile.Triton
abbrev Storage := VeriTile.Bench.TritonBenchG.ContextAttnFwd.Storage
abbrev Layout := VeriTile.Bench.TritonBenchG.ContextAttnFwd.Formula.Layout
abbrev Regions := VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program.Regions

/-- Direct binding to the shared selected source. -/
def originalFP32 (p : Layout) (r : Regions) (scale : ℝ) : ComputeKernel :=
  VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program.kernel p r .fp32 Bool.false scale


def blockEndFactoredFP32 (p : Layout) (r : Regions) (scale : ℝ) : ComputeKernel :=
  blockEndFactoredKernel .fp32 r.query r.key r.value scale r.out r.start r.sequence r.prompt
    p.qRow p.qHead p.qDim p.kBatch p.kHead p.kRow p.kDim
    p.vBatch p.vHead p.vRow p.vDim p.outRow p.outHead p.outDim p.kvGroup p.heads p.D p.BM p.BN


noncomputable def kernel (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) : ComputeKernel :=
  match storage with
  | .fp32 => originalFP32 p r scale
  | .fp16 => VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program.kernel p r .fp16 Bool.false scale
  | .bf16 => VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program.kernel p r .bf16 Bool.false scale

theorem actual_source (p : Layout) (r : Regions) (scale : ℝ) :
    kernel p r .fp32 scale =
      VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program.kernel p r .fp32 Bool.false scale := rfl
end VeriTile.Bench.Repairs.ContextAttnFwd
