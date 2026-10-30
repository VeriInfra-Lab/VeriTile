import bench.tritonbench_g.decay_cumsum.PairIO
import bench.tritonbench_g.decay_cumsum.Proofs.Mathematical_MaskEquivalence

namespace VeriTile.Bench.Repairs.DecayCumsum.SourcePair
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
noncomputable section

/-- All three complete source templates reuse the original proved candidate.
The equality preserves every state and failure, so each selected main/tail
invocation keeps its existing IO, program IDs and runtime-scale binding. -/
theorem equivalent (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) :
    original kind h p scale ≡[assumptions] indexCommuted kind h p scale := by
  cases kind
  · exact VeriTile.Bench.TritonBenchG.DecayCumsum.MaskEquivalence.forward_equiv
      .fp32 "g" "go" h.headStride h.width 1 h.B h.H h.T p.span h.BK h.width scale
  · exact VeriTile.Bench.TritonBenchG.DecayCumsum.MaskEquivalence.prepare_equiv
      .fp32 .fp32 "q" "k" "g" "qg" "kg"
      h.headStride h.width 1 h.B h.H h.T p.span h.BK h.width scale
  · exact VeriTile.Bench.TritonBenchG.DecayCumsum.MaskEquivalence.backward_equiv
      .fp32 .fp32 "dqi" "dqo" "dki" "dko" "q" "k" "g" "dg"
      h.headStride h.width 1 h.B h.H h.T p.span h.BK h.width scale

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.DecayCumsum.SourcePair
