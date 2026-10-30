import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.Host
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.ContextAttnFwd.KernelScale
open VeriTile.Triton
noncomputable section

/-- Exact FP32 scalar ABI values of the ordered Python expression
`1.0 / (D ** 0.5) * 1.4426950408889634`, for the five admitted widths.
This is a launch argument observation, not a numerical kernel-result law. -/
def value : Nat → ℝ
  | 16 => 12102203/33554432
  | 32 => 4278775/16777216
  | 64 => 12102203/67108864
  | 128 => 4278775/33554432
  | 256 => 12102203/134217728
  | _ => 0

/-- Binds the historical abstract scalar interface to the actual kernel ABI. -/
def Bound (c : Host.Call) : Prop := Host.scale c=value c.args.q.dims

#specNonCircular value avoiding [Host.scale]
end
end VeriTile.Bench.Repairs.ContextAttnFwd.KernelScale
