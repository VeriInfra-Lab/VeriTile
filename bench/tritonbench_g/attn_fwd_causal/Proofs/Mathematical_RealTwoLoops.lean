import bench.tritonbench_g.attn_fwd_causal.Proofs.Mathematical_RealSetupSource

/-! Execute the two actual dynamic ranges and the source reset between them.
The entry contract is established by the still-separate prologue proof. The
result is the independent two-range carry, before normalization and storage. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealTwoLoops
open VeriTile.Triton RealLoads RealMemory RealLoopFormula
open RealIteration (Carries Pointers)
open AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 1000000

structure Caller (s : BlockState) (c : Config) (BN BD : Nat) : Prop where
  row : s.regs .nat [] "start_m" = some (Tile.scalar c.startM)
  key : s.regs .ptr [BD,BN] "K_ptrs" = some (keyPointer c BD BN 0)
  scale : s.regs .ptr [] "K_scale_ptr" = some (keyScalePointer c BN 0)
  value : s.regs .ptr [BN,BD] "V_ptrs" = some (valuePointer c BN BD 0)

def writes : List RegName :=
  RealIteration.writes Bool.false ++ RealSetup.writes Bool.true ++ RealIteration.writes Bool.true

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealTwoLoops
