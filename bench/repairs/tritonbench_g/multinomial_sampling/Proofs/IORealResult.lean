import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.LogicalTransport
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.KernelCompatibility

namespace VeriTile.Bench.Repairs.MultinomialSampling.IORealResult
open VeriTile.Triton FP.Structural SourceValue Interface
noncomputable section
variable (rest : Algebra ℝ) (p : Parameters) (pid : Nat)
  (vals : (io rest Real.primitives p pid).Inputs) (s : State ℝ)
  (bound : (io rest Real.primitives p pid).InputBinding vals s) (samePid : s.pids 0=pid)
include bound samePid

theorem weights_eq (i : Row p) (active : SourceValue.active p s i) :
    CDFBridge.weights p s i=weights p vals.memory (pid*p.block+i.1.val) := by
  funext j
  have hr : pid*p.block+i.1.val<p.batches := by
    simpa only [SourceValue.active,SourceValue.row,samePid] using active
  by_cases hj : j<p.tokens
  · have h := IOFacts.score_read rest Real.primitives p pid vals s bound _ j hr hj
    simpa only [CDFBridge.weights,weights,hj,↓reduceIte,SourceValue.row,samePid] using h
  · simp [CDFBridge.weights,weights,hj]

theorem categorical_result (domain : ProbabilityDomain p pid vals.memory) :
    categorical p pid vals.memory
      (nativeReference Real.primitives p pid (data rest Real.primitives p pid vals)) := by
  intro i active
  have ha : SourceValue.active p s i := by
    simpa only [SourceValue.active,SourceValue.row,samePid] using active
  have hw := weights_eq rest p pid vals s bound samePid i ha
  have oldDomain : Spec.Domain p.tokens (CDFBridge.weights p s i) := by
    rw [hw]
    exact domain i active
  obtain ⟨j,pos,value⟩ := Correct.recurrence_result p s i ha oldDomain
  change ∃ j, Spec.Position p.tokens (weights p vals.memory (pid*p.block+i.1.val))
      (Spec.sample (seed p vals.memory (pid*p.block+i.1.val))
        (offset p vals.memory (pid*p.block+i.1.val))) j ∧
      (Logical.rounds Real.primitives p pid (data rest Real.primitives p pid vals) (count p)).output i=
        label p vals.memory (pid*p.block+i.1.val) j
  refine ⟨j,?_,?_⟩
  · have hs := IOFacts.seed_read rest Real.primitives p pid vals s bound _ active
    have ho := IOFacts.offset_read rest Real.primitives p pid vals s bound _ active
    simpa only [hw,SourceValue.row,samePid,hs,ho] using pos
  · have hj : j<max p.tokens 1 := lt_of_lt_of_le pos.1 (Nat.le_max_left _ _)
    have hl := IOFacts.label_read rest Real.primitives p pid vals s bound _ j active hj
    rw [LogicalTransport.rounds_eq rest Real.primitives p pid vals s bound samePid] at value
    exact value.trans (by simpa only [CDFBridge.labels,SourceValue.row,samePid] using hl)

#axiomsClean weights_eq
#axiomsClean categorical_result
end
end VeriTile.Bench.Repairs.MultinomialSampling.IORealResult
