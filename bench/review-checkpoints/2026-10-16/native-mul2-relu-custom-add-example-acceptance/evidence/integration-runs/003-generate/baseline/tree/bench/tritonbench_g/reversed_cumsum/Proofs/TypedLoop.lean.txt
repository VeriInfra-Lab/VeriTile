import bench.tritonbench_g.reversed_cumsum.Proofs.TypedBody
import bench.tritonbench_g.reversed_cumsum.Proofs.FP
import bench.tritonbench_g.reversed_cumsum.Proofs.TypedComposition

/-! Successful complete typed executions in a concrete mathematical algebra.
Both source and candidate have nonempty runs, including real masked accesses,
casts, all reverse iterations and the full final typed state. This supplies a
consistency witness, not a claim about IEEE/native GPU execution. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsum.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite RealProgram
noncomputable section
set_option maxHeartbeats 1800000
set_option maxRecDepth 16000

theorem loop_success (p : Layout) (s o : RegionName) (j : Int)
    (u : State (WithBot ℝ)) (h : Ready p u) :
    ∃ v, rangeInt (engine model) "__rev_i_t" j 1 1
      (TypedProgram.parts Bool.false .bf16 .fp16 p s o).body u = some v ∧ Ready p v := by
  by_cases hj : j < 1
  · obtain ⟨v,he,hv⟩ := body_success p s o u j hj h
    obtain ⟨w,hw,hr⟩ := loop_success p s o (j + 1) v hv
    refine ⟨w,?_,hr⟩
    rw [rangeInt]
    simp only [Nat.one_ne_zero, ↓reduceIte, hj, he, bind, Option.bind_some]
    exact hw
  · refine ⟨u,?_,h⟩
    rw [rangeInt]
    simp only [Nat.one_ne_zero, ↓reduceIte, hj]
termination_by (1 - j).toNat
decreasing_by omega

theorem original_success (p : Layout) (s o : RegionName) (u : State (WithBot ℝ)) :
    ∃ v, FP.Structural.exec (engine model)
      (TypedSource.original .bf16 .fp16 s o p.headStride p.timeStride p.channelStride
        p.T p.S p.BT p.BS) u = some v := by
  obtain ⟨entry,he,hr⟩ := prefix_success Bool.false .bf16 .fp16 p s o u
  obtain ⟨v,hv,_⟩ := loop_success p s o (1 - (numChunks p : Int)) entry hr
  refine ⟨v,?_⟩
  exact TypedComposition.exec_of_split (engine model) _ _ _
    (TypedProgram.source_split Bool.false .bf16 .fp16 p s o) u entry v he
    ((TypedComposition.statement_eval (engine model) p _ entry).trans hv)

theorem nonempty_pair (p : Layout) (s o : RegionName) (u : State (WithBot ℝ)) :
    ∃ v, FP.Structural.exec (engine model)
      (TypedSource.original .bf16 .fp16 s o p.headStride p.timeStride p.channelStride
        p.T p.S p.BT p.BS) u = some v ∧
      FP.Structural.exec (engine model)
        (TypedSource.compareRight .bf16 .fp16 s o p.headStride p.timeStride p.channelStride
          p.T p.S p.BT p.BS) u = some v := by
  obtain ⟨v,hv⟩ := original_success p s o u
  refine ⟨v,hv,?_⟩
  exact (FPProof.whole_state .bf16 .fp16 s o p.headStride p.timeStride
    p.channelStride p.T p.S p.BT p.BS (engine model) u).symm.trans hv

#axiomsClean loop_success
#axiomsClean original_success
#axiomsClean nonempty_pair
end
end VeriTile.Bench.TritonBenchG.ReversedCumsum.TypedExecution
