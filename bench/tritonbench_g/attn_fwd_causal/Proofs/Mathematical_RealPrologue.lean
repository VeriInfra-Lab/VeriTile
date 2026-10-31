import bench.tritonbench_g.attn_fwd_causal.Proofs.Mathematical_RealInit

/-! The complete real source prologue establishes the helper entry contract
from program IDs and actual input memory. No loaded query or carry is assumed.
The only value convention here is zero for masked undefined reads; physical
input windows and output address bounds belong to the final IO contract. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealPrologue
open VeriTile.Triton RealLoads
open AttentionFwdTriton3.RealScores (Frame)
noncomputable section

def configuration (Q K V QS KS : RegionName) (strideM strideD strideKN N HD HA : Nat)
    (v : RealCoordinates.Values) : Config where
  Q := Q
  K := K
  V := V
  QS := QS
  KS := KS
  base := v.base
  queryScaleBase := v.queryScaleBase
  keyScaleBase := v.keyScaleBase
  strideM := strideM
  strideD := strideD
  strideKN := strideKN
  context := N
  headDim := HD
  activeHead := HA
  startM := v.row

def writes : List RegName := RealCoordinates.writes ++ RealPointers.writes ++ RealInit.writes

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealPrologue
