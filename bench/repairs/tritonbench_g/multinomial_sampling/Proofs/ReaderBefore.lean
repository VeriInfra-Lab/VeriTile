import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.IOComplete
import VeriTile.Meta.Specification

/-! Public categorical correctness of the selected complete FP32 source.
IO.lean exposes probability/seed/offset/label windows and original strides;
LogicalIO.lean separately binds arbitrary typed masked-load fallback values.
Operational helpers are in Proofs/; optional Tensor and host results are in Misc/. -/
namespace VeriTile.Bench.Repairs.MultinomialSampling.ReaderBefore
open VeriTile.Triton FP.Structural SourceValue Interface
open scoped BigOperators
noncomputable section

/-- Independent inverse-CDF relation. The sample retains the source Philox10
and its signed offset conversion; cumulative boundaries are mathematical sums. -/
def inverseCDF (p : Parameters) (pid : Nat) (vals : Interface.Inputs ℝ p)
    (outputs : Interface.Outputs p) : Prop :=
  ∀ i : TileIndex [p.block], pid*p.block+i.1.val<p.batches →
    let b := pid*p.block+i.1.val
    let u := Random.randReal (seed p vals b) (BitVec.ofInt 32 (offset p vals b)).toInt .bits32 10
    ∃ j, (j<p.tokens ∧ ((u=0 ∧ j=0) ∨
      (0<u ∧ (∑ k∈Finset.range j, weights p vals b k)<u ∧
        u≤∑ k∈Finset.range (j+1), weights p vals b k))) ∧ outputs i=label p vals b j


/-- Every active row returns its independent inverse-CDF label. The logical
seed condition is exactly the prior signed64 source condition; normalized
probabilities are required only for the active rows of this program. Shared IO
also proves successful execution, actual source safety and the full frame. -/
theorem multinomial_categorical_correctness (rest : Algebra ℝ) (p : Parameters) (pid : Nat) :
    VeriTile.Spec.Real ((io rest Real.primitives p pid).SatisfiesOn
      (fun vals=>Interface.SeedDomain rest Real.primitives p pid vals ∧ ProbabilityDomain p pid vals.memory)
      (fun vals outputs=>inverseCDF p pid vals.memory (outputs (0 : Fin 1)))) :=
  IOComplete.real_satisfies rest p pid

end
end VeriTile.Bench.Repairs.MultinomialSampling.ReaderBefore
