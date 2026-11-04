import bench.tritonbench_g.attn_fwd_triton.Proofs.ActualMiddle
import bench.tritonbench_g.attn_fwd_triton.Proofs.CausalFormula

/-! The caller discards the second maximum but retains exactly the
independent closed numerator and denominator used by the mathematical spec. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualFormula
open VeriTile.Triton Memory
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealLoopFormula (Carry)
open VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

theorem result (mem : BlockState) (c : Config) (BD : Nat) (seed : Carry 128 BD) :
    (ActualMiddle.result mem c BD seed).numerator=
      (TwoLoops.bothResult mem c 128 64 BD (decide (c.p.stage=2)) seed).numerator ∧
    (ActualMiddle.result mem c BD seed).denominator=
      (TwoLoops.bothResult mem c 128 64 BD (decide (c.p.stage=2)) seed).denominator := by
  by_cases hd : c.p.stage=2
  all_goals simp [ActualMiddle.result,ActualCall.result,ActualReturn.returned,
    ActualSetup.lower,ActualSetup.upper,TwoLoops.bothResult,TwoLoops.firstResult,
    FirstSetup.firstLower,FirstSetup.firstUpper,RealLoopFormula.firstLower,hd]

#axiomsClean result
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualFormula
