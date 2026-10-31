import bench.tritonbench_g.attn_fwd_causal.Proofs.Mathematical_RealPreparation
import bench.tritonbench_g.attention_fwd_triton3.Sources.Mathematical_RealLoopFormula

/-! One complete helper iteration, from actual K/V/scale memory loads to the
online carry and all three advanced pointers. The flat body below only selects
the constant first-helper branch. Source transport at the end checks both
original helper bodies. Address bounds and full-kernel IO are separate. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealIteration
open VeriTile.Triton RealLoads RealMemory
open AttentionFwdTriton3.RealTiles (finite pairs dot_finite)
open AttentionFwdTriton3.RealScores (Frame)
open AttentionFwdTriton3.RealLoopFormula (Carry update)
noncomputable section
set_option maxHeartbeats 1200000
set_option linter.unusedSimpArgs false

def updated {BM BN BD : Nat} (xs : RealFormula.Inputs BM BN BD) (c : Config)
    (v : RealFormula.Visit) (carry : Carry BM BD) : Carry BM BD :=
  update carry (RealFormula.logit xs c.startM c.context c.activeHead v) RealTiles.keep
    (fun i => RealFormula.loadedValue xs c.context c.activeHead v i.1 i.2.1)

structure Carries (s : BlockState) {BM BD : Nat} (carry : Carry BM BD) : Prop where
  center : s.regs .real [BM] "m_i" = some (⟨fun i => carry.center i.1⟩ : Tile .real [BM])
  denominator : s.regs .real [BM] "l_i" = some (finite (fun i : TileIndex [BM] => carry.denominator i.1))
  numerator : s.regs .real [BM,BD] "acc" = some (finite carry.numerator)

structure Pointers (s : BlockState) (c : Config) (BN BD start : Nat) (second : Bool) : Prop where
  key : s.regs .ptr [BD,BN] (keyName second) = some (keyPointer c BD BN start)
  scale : s.regs .ptr [] (scaleName second) = some (keyScalePointer c BN start)
  value : s.regs .ptr [BN,BD] (valueName second) = some (valuePointer c BN BD start)

def writes (second : Bool) : List RegName :=
  RealPreparation.writes second ++ ["l_ij","alpha","l_i","acc","v","m_i",
    keyName second,scaleName second,valueName second]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealIteration
