import bench.tritonbench_g.llama_ff_triton.Proofs.SourceSafety
import bench.tritonbench_g.llama_ff_triton.Proofs.IOExecution

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.SourceSafety
open VeriTile.Triton FP.Structural Safety Execution Interface
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem loop_safe (a : Arithmetic α) (native : Bool) (p : Parameters)
    (initial current : State α) (k : Nat) (hk : k≤steps p)
    (h : Invariant a native p initial k current) (bytes : ByteValues initial)
    (bounds : RegionName → Nat) (hb : Bounds p initial bounds) :
    rangeIntSafe a.model bounds "step" k (steps p) 1 (loopBody native p) current := by
  have aux : ∀ n k (current : State α), steps p-k=n → k≤steps p →
      Invariant a native p initial k current →
      rangeIntSafe a.model bounds "step" k (steps p) 1 (loopBody native p) current := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih =>
      intro k current hn hk h
      by_cases hlt : k<steps p
      · let next := bodyExecution a native p initial current k h bytes
        have hi := bodyInvariant a native p initial current k h bytes
        rw [rangeIntSafe_unfold,if_neg (by decide),if_pos (by exact_mod_cast hlt)]
        refine ⟨body_safe a native p initial current k h bytes bounds hb,?_⟩
        rw [next.property]
        simpa only [Int.natCast_add,Int.natCast_one] using
          ih (steps p-(k+1)) (by omega) (k+1) next.val rfl (by omega) hi
      · rw [rangeIntSafe_unfold,if_neg (by decide),if_neg (by omega)]
        trivial
  exact aux _ _ _ rfl hk h

theorem counted_safe (a : Arithmetic α) (native : Bool) (p : Parameters)
    (initial current : State α) (h : Invariant a native p initial 0 current)
    (bytes : ByteValues initial) (bounds : RegionName → Nat) (hb : Bounds p initial bounds) :
    stepSafe a.model bounds
      (.forRangeInt "step" (.constInt 0) countOp (.constNat 1) (loopBody native p)) current := by
  have countSafe : safeOp a.model none bounds current countOp := by
    simp [countOp,safeOp_unfold]
  simpa only [stepSafe_unfold,countSafe,safeOp_unfold,countEvaluation a p current h.k,
    evalOp_unfold,true_and] using loop_safe a native p initial current 0 (by omega) h bytes bounds hb

theorem source_safe (a : Arithmetic α) (p : Parameters) (initial : State α)
    (bytes : ByteValues initial) (bounds : RegionName → Nat) (hb : Bounds p initial bounds) :
    TraceSafe a.model bounds (source p) initial := by
  change traceSafeList a.model bounds (source p).surfaceBody initial
  rw [partition,traceSafeList_append,traceSafeList_append]
  let start := prefixExecution a.model p initial
  have hs := initialInvariant a Bool.false p initial
  obtain ⟨mid,hm,hi⟩ := loopExecution a Bool.false p initial start.val 0 (by omega) hs bytes
  have hmid : FP.Structural.step a.model
      (.forRangeInt "step" (.constInt 0) countOp (.constNat 1) (body p)) start.val=some mid := by
    rw [countedExecution a p start.val _ hs.k]
    simpa [loopBody] using hm
  refine ⟨⟨prelude_safe a.model p initial bounds,?_⟩,?_⟩
  · intro st he
    rw [start.property] at he
    cases Option.some.inj he
    simpa only [traceSafeList_unfold,loopBody,Bool.false_eq_true,ite_false] using
      And.intro (counted_safe a Bool.false p initial start.val hs bytes bounds hb)
        (show (match FP.Structural.step a.model
          (.forRangeInt "step" (.constInt 0) countOp (.constNat 1) (body p)) start.val with
          | some _ => True | none => True) from by split <;> trivial)
  · intro st he
    rw [run_append,start.property] at he
    simp only [Option.bind_some,run,hmid] at he
    cases Option.some.inj he
    exact epilogue_safe a p initial mid hi bounds hb

end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.SourceSafety
