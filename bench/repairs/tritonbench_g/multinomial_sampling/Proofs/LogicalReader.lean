import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.ReaderBefore
import bench.repairs.tritonbench_g.multinomial_sampling.IO
namespace VeriTile.Bench.Repairs.MultinomialSampling.LogicalReader
open VeriTile.Triton FP.Structural SourceValue Interface
open scoped BigOperators
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

theorem judgment_iff (rest : Algebra ℝ) (p : Parameters) (pid : Nat) :
    (LogicalView.samplingIO rest p pid).SatisfiesOn (LogicalView.accepted rest p pid) (inverseCDF rest p pid) ↔
    (io rest Real.primitives p pid).SatisfiesOn
      (fun vals => Interface.SeedDomain rest Real.primitives p pid vals ∧ ProbabilityDomain p pid vals.memory)
      (fun vals outputs => ReaderBefore.inverseCDF p pid vals.memory (outputs (0 : Fin 1))) := by
  unfold LogicalDomainRelationIO.SatisfiesOn
  apply Iff.of_eq
  congr 1
  funext xs outputs
  apply propext
  simp only [LogicalView.samplingIO,inverseCDF,LogicalView.weight,LogicalView.seedValue,
      LogicalView.offsetValue,LogicalView.labelValue,Equiv.symm_apply_apply,LogicalView.outputEquiv,
      Equiv.coe_fn_mk,ReaderBefore.inverseCDF]
  constructor
  · intro h i; exact h i.1
  · intro h i; exact h (i,())
end
end VeriTile.Bench.Repairs.MultinomialSampling.LogicalReader
