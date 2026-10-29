import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPComplete
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPSoftware
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.KernelSource
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.MultinomialSampling.FPEquiv
open VeriTile.Triton FP.Structural SourceValue
noncomputable section

/-- There is one selected implementation. Its FP result is the independent
source-precision recurrence, with all scan/max schedules and primitive nodes
retained. No duplicate implementation is introduced as an equivalence target. -/
theorem exact_kernel_words (rest : Algebra Word) (plans : Plans) (p : Parameters)
    (s : State Word) (seeds : SeedDomain p s) :
    ∃ t,FP.Structural.exec (backend rest (software .fp32 plans)) (kernel p) s=some t ∧
      t.mem=(written (software .fp32 plans) .fp32 p s).mem ∧
      (∀ i : Row p,active p s i → t.mem p.output (row p s i)=
        Cell.mk .int ((rounds (software .fp32 plans) .fp32 p s (count p)).output i)) ∧
      (∀ r a,(r≠p.output ∨ ∀ i : Row p,active p s i → a≠row p s i) → t.mem r a=s.mem r a) :=
  correctness rest (software .fp32 plans) .fp32 p s seeds

/-- The same successful complete execution/readback/frame result holds for
any supported opaque primitives. No algebraic law is used. -/
theorem opaque_kernel_outcome {α : Type} [Inhabited α] (rest : Algebra α)
    (P : Primitives α) (p : Parameters) (s : State α) (seeds : SeedDomain p s) :
    ∃ t,FP.Structural.exec (backend rest P) (kernel p) s=some t ∧
      t.mem=(written P .fp32 p s).mem ∧
      (∀ i : Row p,active p s i → t.mem p.output (row p s i)=
        Cell.mk .int ((rounds P .fp32 p s (count p)).output i)) ∧
      (∀ r a,(r≠p.output ∨ ∀ i : Row p,active p s i → a≠row p s i) → t.mem r a=s.mem r a) :=
  correctness rest P .fp32 p s seeds

#specNonCircular written avoiding [kernel,source,FP.Structural.exec]
#axiomsClean exact_kernel_words
#axiomsClean opaque_kernel_outcome
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MultinomialSampling.FPEquiv
