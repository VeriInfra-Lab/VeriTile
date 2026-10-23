import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Guarded
import bench.repairs.tritonbench_g.multinomial_sampling.Misc.CorrectPublic

namespace VeriTile.Bench.Repairs.MultinomialSampling.Correct
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MultinomialSampling.SourceValue
noncomputable section

/-- Arbitrary normalized Real probabilities; no finite-word representation,
dyadic admission or exact-rounding predicate restricts this theorem. -/
abbrev arbitrary_normalized_real := @Host.whole_real

theorem checked_returned (rest : Algebra ℝ) (api : Physical.API ℝ Store Byte)
    (p : Parameters) (a : Host.Arguments) (s : State ℝ) (call : Native.Preparation api p a s)
    (observed : Admission.Observations p)
    (checked : Admission.check api p a s call observed = Bool.true)
    (fresh : Host.Fresh p a) (undef : Admission.TypedUndefined p s)
    (readback : Admission.Readback p s observed)
    :
    Admission.guarded api Real.primitives p a s call observed =
      some (Native.entry api Real.primitives p a s call) ∧
    (p.tokens ≠ 1 → IndexABI api p a s call) ∧
    ∃ logical, Host.invoke rest Real.primitives p a s = some logical ∧
      logical.shape = (if p.tokens=1 then [p.batches,1] else [p.batches]) ∧
      (Native.entry api Real.primitives p a s call).tensor = call.output ∧
      (Native.entry api Real.primitives p a s call).tensor.shape = logical.shape ∧
      (∀ o, o < p.batches → Host.Result p a s o
        ((Physical.logicalRead api (Native.entry api Real.primitives p a s call).memory
          (Native.entry api Real.primitives p a s call).tensor o).toInt)) ∧
      (∀ b, call.caller b → api.byte (Native.entry api Real.primitives p a s call).memory b = api.byte call.before b) := by
  refine ⟨by simp [Admission.guarded,checked],Admission.checked_abi api p a s call observed checked,?_⟩
  exact whole_native rest api p a s call
    (Admission.checked_valid api p a s call observed checked) fresh
    (Admission.checked_domain api p a s call observed s checked readback)
    (fun _ pid _ => Admission.checked_seeds api p a s call fresh undef pid)

#axiomsClean checked_returned
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MultinomialSampling.Correct
