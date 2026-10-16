import bench.tritonbench_g.multinomial_sampling.Proofs.FPComplete
import bench.tritonbench_g.multinomial_sampling.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.MultinomialSampling.FPCorrect
open VeriTile.Triton FP.Structural FPValue
noncomputable section

/-- Complete original kernel for all four score storage profiles and every
raw token loop length, including tails. Concrete ten-round Philox and original
i64 metadata are retained; scalar floating operations, scan and maximum are
explicit backend primitives. The reference reads initial memory, allows
logical aliases, and preserves last-matching-block overwrite behavior.
SeedDomain is signed64 storage typing, including arbitrary valid masked undef.
This theorem does not assert an ideal probability distribution or GPU behavior. -/
specification multinomial_full_backend_correctness
    {α : Type} [Inhabited α] (rest : Algebra α) (P : Primitives α)
    (d : Storage) (p : Parameters) (initial : State α) (hseed : SeedDomain p initial) :
    ∃ final, FP.Structural.exec (backend rest P) (source d p) initial=some final ∧
      final.mem=(written P d p initial).mem ∧
      (∀ i : Row p, active p initial i → final.mem p.output (row p initial i)=
        .mk .int ((rounds P d p initial (count p)).output i)) ∧
      (∀ r o, (r≠p.output ∨ ∀ i : Row p, active p initial i → o≠row p initial i) →
        final.mem r o=initial.mem r o) := correctness rest P d p initial hseed

/-- Complete software-bit specialization, with actual binary16/bfloat16/
binary32/binary64 score interpretation and conversion, fp32 rounded arithmetic,
ordered comparisons, checked scan/max tree configurations, and concrete RNG.
All NaN/Inf/subnormal/negative inputs remain in scope. Native tree, cast, FTZ,
payload and address/typed-store bindings remain pending. The host's token1
allocation/early-return branch is separate from this complete kernel theorem. -/
specification multinomial_full_software_correctness
    (rest : Algebra Word) (d : Storage) (plans : Plans) (p : Parameters)
    (initial : State Word) (hseed : SeedDomain p initial) :
    ∃ final, FP.Structural.exec (backend rest (software d plans)) (source d p) initial=some final ∧
      final.mem=(written (software d plans) d p initial).mem ∧
      (∀ i : Row p, active p initial i → final.mem p.output (row p initial i)=
        .mk .int ((rounds (software d plans) d p initial (count p)).output i)) ∧
      (∀ r o, (r≠p.output ∨ ∀ i : Row p, active p initial i → o≠row p initial i) →
        final.mem r o=initial.mem r o) :=
  correctness rest (software d plans) d p initial hseed

#axiomsClean multinomial_full_backend_correctness
#axiomsClean multinomial_full_software_correctness
end
end VeriTile.Bench.TritonBenchG.MultinomialSampling.FPCorrect
