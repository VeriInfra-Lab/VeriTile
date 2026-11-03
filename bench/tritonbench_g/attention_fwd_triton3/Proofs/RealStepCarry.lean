import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealStepProbabilities

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Real
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open scoped BigOperators
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000

theorem sum_row (BM BN : Nat) (f : TileIndex [BM,BN] → ℝ) (row : Fin BM) :
    sum (TileShape.axisDim [BM,BN] ⟨1,by simp⟩)
      (reductionInputs ⟨1,by simp⟩ Bool.false (fun i=>some (f i)) (row,()))=
      some (∑ key : Fin BN,f (row,key,())) := by
  change sum BN (fun key=>some (f (row,key,())))=_
  exact sum_some _ _

theorem advance_real (c : Call) (a : Reference.Inputs R) (cur : Int) (old : Reference.Carry c R)
    (seed : RealLoopFormula.Carry c.BM c.D) (encoded : Encodes c old seed)
    (q : TileIndex [c.BM,c.D] → ℝ) (k : TileIndex [c.D,c.BN] → ℝ)
    (v : TileIndex [c.BN,c.D] → ℝ)
    (hq : Reference.query c ops a=fun i=>some (q i))
    (hk : Reference.loadBlock ops a old.keyPointer [c.D,c.BN]
      (if c.params.evenN≠0 then [] else [0,1])=fun i=>some (k i))
    (hv : Reference.loadBlock ops a old.valuePointer [c.BN,c.D]
      (if c.params.evenN≠0 then [] else [0,1])=fun i=>some (v i)) :
    Encodes c (Reference.advance c ops a cur old)
      (RealLoopFormula.update seed (score c q k) (active c a cur) v) := by
  have hs:=scores_real c a cur old q k hq hk
  have hc:=center_real c a cur old seed encoded q k hs
  have hp:=probabilities_real c a cur old seed q k hs hc
  have ha:=alphas_real c a cur old seed encoded q k hc
  unfold Encodes
  rw [Reference.advance_stages]
  refine ⟨hc,?_,?_⟩
  · funext ⟨row,tail⟩
    cases tail
    change (Reference.scaledCarry c ops a cur old).mass (row,())=_
    unfold Reference.scaledCarry
    rw [encoded.2.1,hp,ha]
    simp [ops,algebra,reductionInputs,sum_some,RealLoopFormula.update]
    change Option.map
      (fun b=>seed.denominator row * RealFormula.weight (seed.center row)
        (RealTiles.center (score c q k) (active c a cur) seed.center row)+b)
      (sum c.BN (fun key=>some
        (RealTiles.probability (score c q k) (active c a cur) seed.center row key)))=_
    rw [sum_some]
    dsimp only [Option.map_some]
    rw [RealTiles.sum_probabilities (score c q k) (active c a cur) seed.center row (fun _=>0)]
    rw [RealTiles.center_matches_summary (score c q k) (active c a cur) seed.center row
      (seed.denominator row) 0 (fun _=>0)]
    rfl
  · funext ⟨row,col,tail⟩
    cases tail
    dsimp only
    unfold Reference.scaledCarry
    rw [encoded.2.2,hp,ha,hv]
    simp [ops,algebra,dotValues,ofFloat,toFloat,sum_some,RealLoopFormula.update]
    rw [RealTiles.sum_weighted_values (score c q k) (active c a cur) seed.center row
      (fun key=>v (key,col,()))]
    rw [RealTiles.center_matches_summary (score c q k) (active c a cur) seed.center row
      (seed.denominator row) (seed.numerator (row,col,())) (fun key=>v (key,col,()))]
    rfl

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Real
