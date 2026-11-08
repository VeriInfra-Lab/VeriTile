import bench.tritonbench_g.multinomial_sampling.Proofs.LogicalReader
namespace VeriTile.Bench.Repairs.MultinomialSampling
open VeriTile.Triton FP.Structural SourceValue Interface
open scoped BigOperators VeriTile.Triton.LogicalDomainRelationIO
noncomputable section
/-- Independent inverse-CDF law on every active logical row. The explicit
Philox10 sample retains signed offset conversion; weights use mathematical sums. -/
def inverseCDF (rest : Algebra ℝ) (p : Parameters) (pid : Nat) (xs : LogicalView.Inputs p)
    (outputs : Fin p.block → Int) : Prop :=
  ∀ lane : Fin p.block, pid*p.block+lane.val<p.batches →
    let b := pid*p.block+lane.val
    let u := Random.randReal (LogicalView.seedValue rest p pid xs b)
      (BitVec.ofInt 32 (LogicalView.offsetValue rest p pid xs b)).toInt .bits32 10
    ∃ j, (j<p.tokens ∧ ((u=0 ∧ j=0) ∨
      (0<u ∧ (∑ k∈Finset.range j, LogicalView.weight rest p pid xs b k)<u ∧
        u≤∑ k∈Finset.range (j+1), LogicalView.weight rest p pid xs b k))) ∧
      outputs lane=LogicalView.labelValue rest p pid xs b j

/-- Signed64 seed/fallback range and normalized nonnegative probabilities are
the exact original domain. Undefined inactive-load inputs remain arbitrary.
The source safety, success and complete raw exterior frame are preserved. -/
specification multinomial_categorical_correctness (rest : Algebra ℝ) (p : Parameters) (pid : Nat) :
    VeriTile.Spec.Real (LogicalView.samplingIO rest p pid ⊨ᵣ{LogicalView.accepted rest p pid} inverseCDF rest p pid) :=
  (LogicalReader.judgment_iff rest p pid).mpr (ReaderBefore.multinomial_categorical_correctness rest p pid)
#axiomsClean multinomial_categorical_correctness
end
end VeriTile.Bench.Repairs.MultinomialSampling
