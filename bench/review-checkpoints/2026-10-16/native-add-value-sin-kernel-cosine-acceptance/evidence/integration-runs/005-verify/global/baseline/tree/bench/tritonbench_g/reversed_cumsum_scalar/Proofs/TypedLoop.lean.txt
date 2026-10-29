import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.TypedBody
import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.FP
import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.TypedComposition

/-! Successful complete typed executions in the same profiled interpreter
as the public floating equivalence. This mathematical consistency witness
includes real scan/reduction computations; it does not assert IEEE refinement. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite RealProgram
noncomputable section
set_option maxHeartbeats 1800000
set_option maxRecDepth 16000

theorem loop_success (T BT : Nat) (s o : RegionName) (j : Int)
    (u : State (WithBot ℝ)) (h : Ready u) :
    ∃ v, rangeInt (engine model) "__rev_i_t" j 1 1
      (TypedProgram.parts Bool.false .bf16 .fp16 s o T BT).body u = some v ∧ Ready v := by
  by_cases hj : j < 1
  · obtain ⟨v,he,hv⟩ := body_success T BT s o u j hj h
    obtain ⟨w,hw,hr⟩ := loop_success T BT s o (j + 1) v hv
    refine ⟨w,?_,hr⟩
    rw [rangeInt]
    simp only [Nat.one_ne_zero, ↓reduceIte, hj, he, bind, Option.bind_some]
    exact hw
  · refine ⟨u,?_,h⟩
    rw [rangeInt]
    simp only [Nat.one_ne_zero, ↓reduceIte, hj]
termination_by (1 - j).toNat
decreasing_by omega

theorem original_success (T BT : Nat) (s o : RegionName) (u : State (WithBot ℝ)) :
    ∃ v, FP.Structural.exec (engine model) (TypedSource.original .bf16 .fp16 s o T BT) u = some v := by
  obtain ⟨entry,he,hr⟩ := prefix_success .bf16 .fp16 T BT s o u
  obtain ⟨v,hv,_⟩ := loop_success T BT s o (1 - (numChunks T BT : Int)) entry hr
  refine ⟨v,?_⟩
  exact TypedComposition.exec_of_split (engine model) _ _ _
    (TypedProgram.source_split Bool.false .bf16 .fp16 s o T BT) u entry v he
    ((TypedComposition.statement_eval (engine model) T BT _ entry).trans hv)

theorem successful_pair (T BT : Nat) (s o : RegionName) (u : State (WithBot ℝ)) :
    ∃ v, FP.Structural.exec (engine model) (TypedSource.original .bf16 .fp16 s o T BT) u = some v ∧
      FP.Structural.exec (engine model) (TypedSource.indexCommuted .bf16 .fp16 s o T BT) u = some v := by
  obtain ⟨v,hv⟩ := original_success T BT s o u
  refine ⟨v,hv,?_⟩
  exact (FPProof.whole_state .bf16 .fp16 s o T BT (engine model) u).symm.trans hv

/-- Empty registers, two chunks, a partial final chunk, independent source
and output dtypes, nonzero row ID and a nonzero unspecified padding oracle. -/
theorem nonempty_pair :
    ∃ v, FP.Structural.exec (engine model) (TypedSource.original .bf16 .fp16 "s" "o" 17 16) initial = some v ∧
      FP.Structural.exec (engine model) (TypedSource.indexCommuted .bf16 .fp16 "s" "o" 17 16) initial = some v :=
  successful_pair 17 16 "s" "o" initial

#axiomsClean loop_success
#axiomsClean original_success
#axiomsClean successful_pair
#axiomsClean nonempty_pair
end
end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.TypedExecution
