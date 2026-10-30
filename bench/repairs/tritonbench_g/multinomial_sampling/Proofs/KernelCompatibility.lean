import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.CDFBridge
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPComplete
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.KernelMemory
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.MultinomialSampling.Correct
open VeriTile.Triton FP.Structural SourceValue
noncomputable section

/-- Independent inverse-CDF output relation on the initial row. The sample is
canonical rounded Philox10, with signed32 offset narrowing. Zero chooses the
first label; positive samples use strict-lower/inclusive-upper intervals. -/
def Result (p : Parameters) (s : State ℝ) (i : Row p) (output : Int) : Prop :=
  Spec.Result p.tokens (CDFBridge.weights p s i) (CDFBridge.labels p s i)
    ((s.mem p.seeds (row p s i)).read .int)
    ((s.mem p.offsets (row p s i)).read .int) output

theorem recurrence_result (p : Parameters) (s : State ℝ) (i : Row p)
    (ha : active p s i) (domain : Spec.Domain p.tokens (CDFBridge.weights p s i)) :
    Result p s i ((rounds Real.primitives .fp32 p s (count p)).output i) := by
  obtain ⟨j,hj,hout⟩ := CDFBridge.rounds_result p s i ha domain
  refine ⟨j,?_,hout⟩
  have sample_eq : sample Real.primitives p s i =
      Spec.sample ((s.mem p.seeds (row p s i)).read .int)
        ((s.mem p.offsets (row p s i)).read .int) := by
    rw [Real.primitive_sample]
    simp [Spec.sample,seedValue,offsetValue,intLoad,ha]
  simpa only [sample_eq] using hj

/-- Successful execution of the complete selected kernel in Real arithmetic,
with independent categorical output and the entire exterior typed-memory
frame. Input probabilities are arbitrary normalized Reals, and the theorem
assumes no FP exactness or successful execution. -/
theorem multinomial_kernel_correctness (rest : Algebra ℝ) (p : Parameters)
    (s : State ℝ) (seeds : SeedDomain p s)
    (domain : ∀ i : Row p,active p s i → Spec.Domain p.tokens (CDFBridge.weights p s i)) :
    VeriTile.Spec.Real (∃ t,
      FP.Structural.exec (backend rest Real.primitives) (kernel p) s=some t ∧
      (∀ i : Row p,active p s i → Result p s i ((t.mem p.output (row p s i)).read .int)) ∧
      (∀ r a,(r≠p.output ∨ ∀ i : Row p,active p s i → a≠row p s i) → t.mem r a=s.mem r a)) := by
  obtain ⟨t,he,_,hv,hf⟩ := correctness rest Real.primitives .fp32 p s seeds
  refine ⟨t,he,?_,hf⟩
  intro i ha
  rw [hv i ha,Cell.read_same]
  exact recurrence_result p s i ha (domain i ha)

abbrev output_coverage := @KernelMemory.output_coverage

#specNonCircular Result avoiding [kernel,source,FP.Structural.exec,rounds,written]
#axiomsClean recurrence_result
#axiomsClean multinomial_kernel_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MultinomialSampling.Correct
