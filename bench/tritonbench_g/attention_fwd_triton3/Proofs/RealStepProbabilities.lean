import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealStepScores

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Real
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000

theorem exp2_difference (a b : R) :
    exp2 (WithBot.realSub a b)=some (RealFormula.weight a b) := by
  cases a <;> cases b <;> rfl

theorem exp2_map_difference (a : ℝ) (b : R) :
    exp2 (b.map (fun x=>a-x))=some (RealFormula.weight (some a) b) := by
  cases b <;> rfl

theorem alphas_real (c : Call) (a : Reference.Inputs R) (cur : Int) (old : Reference.Carry c R)
    (seed : RealLoopFormula.Carry c.BM c.D) (encoded : Encodes c old seed)
    (q : TileIndex [c.BM,c.D] → ℝ) (k : TileIndex [c.D,c.BN] → ℝ)
    (hc : Reference.nextCenter c ops a cur old=
      fun i=>RealTiles.center (score c q k) (active c a cur) seed.center i.1) :
    Reference.alphas c ops a cur old=
      fun i=>some (RealFormula.weight (seed.center i.1)
        (RealTiles.center (score c q k) (active c a cur) seed.center i.1)) := by
  unfold Reference.alphas
  rw [encoded.1,hc]
  funext ⟨row,tail⟩
  cases tail
  simp [ops,algebra,exp2_difference]
  exact exp2_difference _ _

theorem probabilities_real (c : Call) (a : Reference.Inputs R) (cur : Int) (old : Reference.Carry c R)
    (seed : RealLoopFormula.Carry c.BM c.D)
    (q : TileIndex [c.BM,c.D] → ℝ) (k : TileIndex [c.D,c.BN] → ℝ)
    (hs : Reference.scores c ops a cur old=
      fun i=>if active c a cur i.1 i.2.1 then some (score c q k i.1 i.2.1) else none)
    (hc : Reference.nextCenter c ops a cur old=
      fun i=>RealTiles.center (score c q k) (active c a cur) seed.center i.1) :
    Reference.probabilities c ops a cur old=
      fun i=>some (RealTiles.probability (score c q k) (active c a cur) seed.center i.1 i.2.1) := by
  unfold Reference.probabilities
  rw [hs,hc]
  funext ⟨row,key,tail⟩
  cases tail
  by_cases hw : c.window=0 <;> by_cases he : c.params.evenN=0
  all_goals cases hm : Reference.windowMask c a cur (row,key,())
  all_goals by_cases ht : (key.val : Int)+cur<c.NK
  all_goals simp [active,RealTiles.probability,hw,he,hm,ht,ops,algebra,exp2_difference]
  all_goals exact exp2_map_difference _ _

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Real
