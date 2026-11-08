import bench.tritonbench_g.attn_fwd_triton.Proofs.HelperRegisters

/-! Transport the independent online loop theorem to the actual hygienic
helper body. Bounds, stride, failures and the full register frame are kept. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.HelperLoops
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealScores (Frame)
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealLoopFormula (Carry)
open RegisterViews HelperRegisters
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 3000000

theorem executes (mem s : BlockState) (c : Memory.Config)
    (BM BN BD lower upper : Nat) (second : Bool) (seed : Carry BM BD)
    (positive : 0<BN)
    (ready : Loop.Registers mem (pull (names second) s) c BM BN BD lower second seed) :
    ∃ t, stepForRangeIntAux (names second "start_n") (lower : Int) (upper : Int) BN
        (HelperRegisters.body BM BN BD c.p second) s=some t ∧
      Loop.Registers mem (pull (names second) t) c BM BN BD
        (Loop.cursor lower BN (Loop.count lower upper BN)) second
        (Loop.after mem c BM BN BD lower second (decide (c.p.stage=2)) seed
          (Loop.count lower upper BN)) ∧
      Frame (Iteration.writes second) (pull (names second) s) (pull (names second) t) := by
  obtain ⟨old,ran,registers,frame⟩ := Loop.executes mem (pull (names second) s) c
    BM BN BD lower upper second (decide (c.p.stage=2)) seed positive ready
  have matched := RegisterViews.signed_range (names second) "start_n"
    (sourceIteration BM BN BD c.p.headDim second (decide (c.p.stage=2)))
    (HelperRegisters.body BM BN BD c.p second)
    (iteration_corresponds BM BN BD c.p second) (lower : Int) (upper : Int) BN s
  rw [ran] at matched
  cases actual : stepForRangeIntAux (names second "start_n") (lower : Int) (upper : Int) BN
      (HelperRegisters.body BM BN BD c.p second) s with
  | none=>simp only [actual,Option.map_none] at matched; cases matched
  | some t=>
    have same : pull (names second) t=old := Option.some.inj
      (by simpa only [actual,Option.map_some] using matched)
    refine ⟨t,rfl,?_,?_⟩
    · simpa only [same] using registers
    · simpa only [same] using frame

#axiomsClean executes
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.HelperLoops
