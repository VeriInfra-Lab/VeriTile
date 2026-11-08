import bench.tritonbench_g.context_attn_nopad.Proofs.Sources.Mathematical_HostConfig

/-! Complete selected FP32 source in the accepted bounded machine-index
projection. Its signed-width transport is unchanged. -/
namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly
open VeriTile.Triton
abbrev Call := VeriTile.Bench.TritonBenchG.ContextAttnNopad.Host.Call
abbrev Layout := VeriTile.Bench.TritonBenchG.ContextAttnNopad.Formula.Layout
abbrev Regions := VeriTile.Bench.TritonBenchG.ContextAttnNopad.Program.Regions

/-- Direct binding to the shared selected source. -/
def originalFP32 (p : Layout) (r : Regions) (scale : ℝ) : ComputeKernel :=
  VeriTile.Bench.TritonBenchG.ContextAttnNopad.Program.kernel p r .fp32 Bool.false scale


def blockStartReusedFP32 (p : Layout) (r : Regions) (scale : ℝ) : ComputeKernel :=
  blockStartReusedKernel .fp32 r.query r.key r.value scale r.start r.sequence r.out
    p.qRow p.qHead p.qDim p.kRow p.kHead p.kDim p.vRow p.vHead p.vDim
    p.outRow p.outHead p.outDim p.BM p.D p.BN


noncomputable def kernel (c : Call) (r : Regions) : ComputeKernel := originalFP32 c.layout r c.scale
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly
