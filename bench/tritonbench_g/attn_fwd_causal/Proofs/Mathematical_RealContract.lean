import bench.tritonbench_g.attn_fwd_causal.Proofs.Mathematical_RealIO
import bench.tritonbench_g.attn_fwd_causal.Proofs.Mathematical_RealSafetyMiddle
import bench.tritonbench_g.attn_fwd_causal.Proofs.Mathematical_RealAccessSafety

/-! Full real IO correctness for both complete source variants. The expected
value is the independent weighted sum over actual helper visits. The source's
unused stride arguments remain visible; Q strides still control V/Out access.
Physical allocation bounds and input pins are ordinary contract premises. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealContract
open VeriTile.Triton RealLoads
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

structure Parameters where
  (Q K V QS KS Out : RegionName)
  (queryBatch queryHead queryRow queryDim : Nat)
  (keyBatch keyHead keyRow keyDim : Nat)
  (valueBatch valueHead valueRow valueDim : Nat)
  (outputBatch outputHead outputRow outputDim : Nat)
  (batches heads context headDim activeHead stage : Nat)
  validStage : stage = 2 ∨ stage = 3

def configuration (BM BN : Nat) (p : Parameters) (startM headBatch : Nat) : Config where
  Q := p.Q
  K := p.K
  V := p.V
  QS := p.QS
  KS := p.KS
  base := headBatch/p.heads*p.queryBatch+headBatch%p.heads*p.queryHead
  queryScaleBase := headBatch*RealCoordinates.ceil p.context BM
  keyScaleBase := headBatch*RealCoordinates.ceil p.context BN
  strideM := p.queryRow
  strideD := p.queryDim
  strideKN := p.keyRow
  headDim := p.headDim
  context := p.context
  activeHead := p.activeHead
  startM := startM

def expected (BM BN BD startM : Nat) (p : Parameters) (xs : RealFormula.Inputs BM BN BD)
    (i : TileIndex [BM,BD]) : ℝ :=
  RealFormula.expected xs startM p.context p.activeHead p.stage i.1 i.2.1

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealContract
