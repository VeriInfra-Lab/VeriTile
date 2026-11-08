import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealBackend
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealLoopFormula

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Real
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open scoped BigOperators
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000

def score (c : Call) (q : TileIndex [c.BM,c.D] → ℝ) (k : TileIndex [c.D,c.BN] → ℝ)
    (row : Fin c.BM) (key : Fin c.BN) : ℝ :=
  (∑ d : Fin c.D,q (row,d,())*k (d,key,()))*(c.scale*1.4426950408889634)

def active (c : Call) (a : Reference.Inputs R) (cur : Int) (row : Fin c.BM) (key : Fin c.BN) : Bool :=
  (if c.window≠0 then Reference.windowMask c a cur (row,key,()) else Bool.true) &&
    (if c.params.evenN=0 then decide ((key.val : Int)+cur<c.NK) else Bool.true)

def Encodes (c : Call) (old : Reference.Carry c R) (seed : RealLoopFormula.Carry c.BM c.D) : Prop :=
  old.center=(fun i=>seed.center i.1) ∧
  old.mass=(fun i=>some (seed.denominator i.1)) ∧
  old.weighted=(fun i=>some (seed.numerator i))

theorem scores_real (c : Call) (a : Reference.Inputs R) (cur : Int) (old : Reference.Carry c R)
    (q : TileIndex [c.BM,c.D] → ℝ) (k : TileIndex [c.D,c.BN] → ℝ)
    (hq : Reference.query c ops a=fun i=>some (q i))
    (hk : Reference.loadBlock ops a old.keyPointer [c.D,c.BN]
      (if c.params.evenN≠0 then [] else [0,1])=fun i=>some (k i)) :
    Reference.scores c ops a cur old=
      fun i=>if active c a cur i.1 i.2.1 then some (score c q k i.1 i.2.1) else none := by
  unfold Reference.scores
  rw [hq,hk]
  funext ⟨row,key,tail⟩
  cases tail
  by_cases hw : c.window=0 <;> by_cases he : c.params.evenN=0
  all_goals simp [score,active,Reference.scale,ops,algebra,dotValues,
    sum_some,ofFloat,toFloat,hw,he,WithBot.realAdd_coe_coe,WithBot.realMul_coe_coe]
  all_goals first | rfl | skip
  all_goals cases hm : Reference.windowMask c a cur (row,key,())
  all_goals by_cases ht : (key.val : Int)+cur<c.NK
  all_goals simp [hm,ht]

theorem center_real (c : Call) (a : Reference.Inputs R) (cur : Int) (old : Reference.Carry c R)
    (seed : RealLoopFormula.Carry c.BM c.D) (encoded : Encodes c old seed)
    (q : TileIndex [c.BM,c.D] → ℝ) (k : TileIndex [c.D,c.BN] → ℝ)
    (hs : Reference.scores c ops a cur old=
      fun i=>if active c a cur i.1 i.2.1 then some (score c q k i.1 i.2.1) else none) :
    Reference.nextCenter c ops a cur old=
      fun i=>RealTiles.center (score c q k) (active c a cur) seed.center i.1 := by
  unfold Reference.nextCenter
  rw [hs,encoded.1]
  funext ⟨row,tail⟩
  cases tail
  simp [RealTiles.center,RealTiles.maxScore_pairs,
    ops,algebra,reductionInputs]
  rfl

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Real
