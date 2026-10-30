import bench.repairs.tritonbench_g.decay_cumsum.Kernels

namespace VeriTile.Bench.Repairs.DecayCumsum.LegacySurface
open VeriTile.Triton
noncomputable section

def source (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) : ComputeKernel :=
  match kind with
  | .forward => Original.forward .fp32 Bool.false "g" "go"
      h.headStride h.width 1 h.B h.H h.T p.span h.BK h.width scale
  | .prepare => Original.prepare .fp32 .fp32 Bool.false "q" "k" "g" "qg" "kg"
      h.headStride h.width 1 h.B h.H h.T p.span h.BK h.width scale
  | .backward => Original.backward .fp32 .fp32 Bool.false
      "dqi" "dqo" "dki" "dko" "q" "k" "g" "dg"
      h.headStride h.width 1 h.B h.H h.T p.span h.BK h.width scale

/-- The existing mask-index candidate, specialized to the same packed FP32
main/tail source as `source`. Only `i_k * BK` in the lane mask is commuted. -/
def indexCommuted (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) : ComputeKernel :=
  match kind with
  | .forward => Original.forward .fp32 Bool.true "g" "go"
      h.headStride h.width 1 h.B h.H h.T p.span h.BK h.width scale
  | .prepare => Original.prepare .fp32 .fp32 Bool.true "q" "k" "g" "qg" "kg"
      h.headStride h.width 1 h.B h.H h.T p.span h.BK h.width scale
  | .backward => Original.backward .fp32 .fp32 Bool.true
      "dqi" "dqo" "dki" "dko" "q" "k" "g" "dg"
      h.headStride h.width 1 h.B h.H h.T p.span h.BK h.width scale


end
end VeriTile.Bench.Repairs.DecayCumsum.LegacySurface
