import bench.tritonbench_g.context_attn_bloom.Proofs.RealIO
import VeriTile.Triton.Float.ExecutionProfile
import VeriTile.Triton.Float.Control
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnBloom
open Math Program RealSetup
abbrev Layout := Math.Layout
abbrev Regions := Program.Regions
noncomputable def kernel (p : Layout) (r : Regions) (scale : ℝ) : ComputeKernel :=
  Program.kernel p r .fp32 Bool.false scale
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001
