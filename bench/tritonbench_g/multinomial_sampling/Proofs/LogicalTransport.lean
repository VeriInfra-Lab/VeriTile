import bench.tritonbench_g.multinomial_sampling.Proofs.IOFacts

namespace VeriTile.Bench.Repairs.MultinomialSampling.LogicalTransport
open VeriTile.Triton FP.Structural SourceValue Interface
noncomputable section
variable {α : Type} [Inhabited α]
variable (rest : Algebra α) (P : Primitives α) (p : Parameters) (pid : Nat)
  (vals : (io rest P p pid).Inputs) (s : State α)
  (bound : (io rest P p pid).InputBinding vals s) (samePid : s.pids 0=pid)
include bound samePid
local notation "x" => data rest P p pid vals

theorem scores_eq (b : Nat) :
    SourceValue.scores P .fp32 p s b=Logical.scores P p pid x b := by
  funext i
  simp only [SourceValue.scores,Logical.scores,readScore,Storage.dtype,SourceValue.active,
    SourceValue.row,scoreOffset,Logical.row,IOUndef.row,samePid]
  split_ifs with h
  · exact congrArg (P.cast (some .fp32) .real .real)
      (IOFacts.score_read rest P p pid vals s bound _ _ h.1 h.2)
  · rfl

theorem sample_eq : SourceValue.sample P p s=Logical.sample P p pid x := by
  funext i
  simp only [SourceValue.sample,Logical.sample,
    IOFacts.seed_value rest P p pid vals s bound samePid,
    IOFacts.offset_value rest P p pid vals s bound samePid]

theorem valid_eq (b : Nat) (st : Accumulator α p) :
    SourceValue.valid P .fp32 p s b st=Logical.valid P p pid x b st := by
  funext i
  simp only [SourceValue.valid,SourceValue.cumulative,SourceValue.scanned,
    Logical.valid,Logical.cumulative,Logical.scanned,
    scores_eq rest P p pid vals s bound samePid,sample_eq rest P p pid vals s bound samePid]

/-- Transport the already proved source recurrence to finite logical inputs.
The proof retains all lanes, including arbitrary masked integer payloads. -/
theorem rounds_eq (k : Nat) :
    SourceValue.rounds P .fp32 p s k=Logical.rounds P p pid x k := by
  induction k with
  | zero =>
    unfold SourceValue.rounds SourceValue.initialAccumulator Logical.rounds Logical.initial
    congr 1
    funext i
    exact IOFacts.initial_label rest P p pid vals s bound samePid i
  | succ k ih =>
    unfold SourceValue.rounds Logical.rounds
    rw [ih]
    unfold SourceValue.advance Logical.advance
    congr 1
    · funext i
      simp only [SourceValue.scanned,Logical.scanned,scores_eq rest P p pid vals s bound samePid]
    · funext i
      have hv := valid_eq rest P p pid vals s bound samePid (k*p.blockTokens)
        (Logical.rounds P p pid x k)
      have hfound : SourceValue.found P .fp32 p s (k*p.blockTokens) (Logical.rounds P p pid x k)=
          Logical.found P p pid x (k*p.blockTokens) (Logical.rounds P p pid x k) := by
        funext j
        unfold SourceValue.found Logical.found
        rw [hv]
      have hselected : SourceValue.selected P .fp32 p s (k*p.blockTokens) (Logical.rounds P p pid x k)=
          Logical.selected P p pid x (k*p.blockTokens) (Logical.rounds P p pid x k) := by
        funext j
        unfold SourceValue.selected Logical.selected
        rw [hv]
      rw [hfound]
      by_cases hf : Logical.found P p pid x (k*p.blockTokens) (Logical.rounds P p pid x k) i=Bool.true
      · simp only [hf,↓reduceIte,SourceValue.selectedIndex,hfound,Bool.true_and,hselected]
        by_cases ha : SourceValue.active p s i
        · have har : Logical.row p pid i<p.batches := by simpa only [SourceValue.active,SourceValue.row,Logical.row,IOUndef.row,samePid] using ha
          have harRaw : pid*p.block+i.1.val<p.batches := har
          have hj : min (Logical.selected P p pid x (k*p.blockTokens) (Logical.rounds P p pid x k) i)
              (p.tokens-1)<max p.tokens 1 := by omega
          have hl := IOFacts.label_read rest P p pid vals s bound _ _ har hj
          simpa [ha,har,harRaw,Logical.row,IOUndef.row,SourceValue.row,samePid] using hl
        · have har : ¬Logical.row p pid i<p.batches := by simpa only [SourceValue.active,SourceValue.row,Logical.row,IOUndef.row,samePid] using ha
          simp [ha,har]
      · simp [hf]

#axiomsClean scores_eq
#axiomsClean sample_eq
#axiomsClean rounds_eq
end
end VeriTile.Bench.Repairs.MultinomialSampling.LogicalTransport
