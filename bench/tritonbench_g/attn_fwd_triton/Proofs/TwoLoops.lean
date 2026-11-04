import bench.tritonbench_g.attn_fwd_triton.Proofs.FirstSetup

/-! Independent closed carry formulas for the two source helper ranges.
Complete source execution is proved in ActualMiddle; no inline kernel is kept here. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.TwoLoops
open VeriTile.Triton Memory
open VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealScores (Frame)
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealLoopFormula (Carry)
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def firstResult (mem : BlockState) (c : Config) (BM BN BD : Nat) (diagonal : Bool) (seed : Carry BM BD) : Carry BM BD :=
  Loop.after mem c BM BN BD (FirstSetup.firstLower c.startM BM diagonal) Bool.false diagonal seed
    (Loop.count (FirstSetup.firstLower c.startM BM diagonal) (FirstSetup.firstUpper c.startM BM c.p.context diagonal) BN)

def bothResult (mem : BlockState) (c : Config) (BM BN BD : Nat) (diagonal : Bool) (seed : Carry BM BD) : Carry BM BD :=
  Loop.after mem c BM BN BD (c.startM*BM) Bool.true diagonal (firstResult mem c BM BN BD diagonal seed)
    (Loop.count (c.startM*BM) (min ((c.startM+1)*BM) c.p.context) BN)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.TwoLoops
