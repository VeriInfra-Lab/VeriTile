import bench.tritonbench_g.attn_fwd_causal.Proofs.Mathematical_RealMiddle

/-! The eight leading coordinate/base assignments. Source integer address
arithmetic uses the DSL's Nat channel; launch bounds and address validity are
separate contracts. The unused vk_offset assignment is retained and executed. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealCoordinates
open VeriTile.Triton
open AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 800000

def ceil (N B : Nat) : Nat := (N+B-1)/B
def ceilExpr (N B : Nat) : Op .nat [] :=
  .div .nat .nil (.sub .nat .nil (.add .nat .nil (.constNat N) (.constNat B)) (.constNat 1)) (.constNat B)

def writes : List RegName := ["start_m","off_hz","off_z","off_h","qvk_offset","vk_offset","q_scale_offset","k_scale_offset"]

structure Values where
  (row headBatch batch head base valueBase queryScaleBase keyScaleBase : Nat)

def values (s : BlockState) (H N BM BN strideZ strideH strideM : Nat) : Values where
  row := s.pids 0
  headBatch := s.pids 1
  batch := s.pids 1/H
  head := s.pids 1%H
  base := (s.pids 1/H)*strideZ+(s.pids 1%H)*strideH
  valueBase := ((s.pids 1/H)*strideZ+(s.pids 1%H)*strideH)/strideM
  queryScaleBase := s.pids 1*ceil N BM
  keyScaleBase := s.pids 1*ceil N BN

structure Registers (v : Values) (s : BlockState) : Prop where
  row : s.regs .nat [] "start_m" = some (Tile.scalar v.row)
  headBatch : s.regs .nat [] "off_hz" = some (Tile.scalar v.headBatch)
  batch : s.regs .nat [] "off_z" = some (Tile.scalar v.batch)
  head : s.regs .nat [] "off_h" = some (Tile.scalar v.head)
  base : s.regs .nat [] "qvk_offset" = some (Tile.scalar v.base)
  valueBase : s.regs .nat [] "vk_offset" = some (Tile.scalar v.valueBase)
  queryScaleBase : s.regs .nat [] "q_scale_offset" = some (Tile.scalar v.queryScaleBase)
  keyScaleBase : s.regs .nat [] "k_scale_offset" = some (Tile.scalar v.keyScaleBase)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealCoordinates
