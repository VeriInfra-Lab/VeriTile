import bench.tritonbench_g.attn_fwd_causal.Proofs.Mathematical_RealMiddleSource

/-! Complete execution from the loaded query/scale and initial carry through
both helpers. The caller pointer contract precedes local helper setup; no
loop-output or successful-execution premise is required. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealMiddle
open VeriTile.Triton RealLoads RealLoopFormula
open AttentionFwdTriton3.RealScores (Frame)
noncomputable section

structure Entry {BM BN BD : Nat} (xs : RealFormula.Inputs BM BN BD) (c : Config) (s : BlockState) : Prop where
  query : s.regs .real [BM,BD] "q" = some (RealTiles.query xs c.startM c.context c.activeHead)
  scale : s.regs .real [] "q_scale" = some (Tile.scalar (some xs.queryScale))
  rows : s.regs .nat [BM] "offs_m" = some ⟨fun i => c.startM*BM+i.1.val⟩
  keys : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun i => i.val))
  carry : RealIteration.Carries s (initial BM BD)
  caller : RealTwoLoops.Caller s c BN BD
  undef : ∀ r a, s.undef r a = 0

def writes : List RegName := RealSetup.writes Bool.false ++ RealTwoLoops.writes

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealMiddle
