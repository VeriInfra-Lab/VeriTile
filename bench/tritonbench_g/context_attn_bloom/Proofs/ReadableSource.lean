import bench.tritonbench_g.context_attn_bloom.Kernels
import bench.tritonbench_g.context_attn_bloom.Proofs.Kernels

namespace VeriTile.Bench.Repairs.ContextAttnBloomSource
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnBloom
noncomputable section

/-- Entire source-function identity: all regions, strides, shapes and scale
arguments remain universally quantified through this equality. -/
theorem original_preserves : kernel=original Storage.fp32 := rfl

abbrev SourceFunction := RegionName → RegionName → RegionName → ℝ → RegionName → RegionName → RegionName → RegionName → RegionName → RegionName → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → ComputeKernel

def atLayout (source : SourceFunction) (p : Math.Layout) (r : Program.Regions) (scale : ℝ) : ComputeKernel :=
  source r.query r.key r.value scale r.start r.sequence r.out r.tokens r.request r.prompt
    p.qRow p.qHead p.qDim p.kRow p.kHead p.kDim p.vRow p.vHead p.vDim
    p.outRow p.outHead p.outDim p.requestRow p.requestToken p.kvGroup p.headDim p.BM p.D p.BN

theorem selected_binding (p : Math.Layout) (r : Program.Regions) (scale : ℝ) :
    VeriTile.Bench.Repairs.ContextAttnBloomWhole001.kernel p r scale=atLayout kernel p r scale := rfl

#axiomsClean original_preserves
#axiomsClean selected_binding
end
end VeriTile.Bench.Repairs.ContextAttnBloomSource
