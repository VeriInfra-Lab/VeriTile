import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Bits
import bench.repairs.tritonbench_g.multinomial_sampling.Misc.FPEquivPublic

namespace VeriTile.Bench.Repairs.MultinomialSampling.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MultinomialSampling.SourceValue
noncomputable section

theorem checked_returned (rest : Algebra Word) (api : Physical.API Word Store Byte) (plans : Plans)
    (p : Parameters) (a : Host.Arguments) (word : State Word) (real : State ℝ)
    (call : Native.Preparation api p a word)
    (observed : Admission.Observations p)
    (checked : Admission.check api p a word call observed = Bool.true)
    (fresh : Host.Fresh p a) (undef : Admission.TypedUndefined p word)
    (inputs : Numerics.Inputs p word real)
    (readback : Admission.Readback p real observed)
    (nodes : p.tokens ≠ 1 → Numerics.LaunchExact plans p a word)
    :
    Admission.guarded api (software .fp32 plans) p a word call observed =
      some (Native.entry api (software .fp32 plans) p a word call) ∧
    (p.tokens ≠ 1 → IndexABI api p a word call) ∧
    ∃ logical, Host.invoke rest (software .fp32 plans) p a word = some logical ∧
      logical.shape = (if p.tokens=1 then [p.batches,1] else [p.batches]) ∧
      (Native.entry api (software .fp32 plans) p a word call).tensor = call.output ∧
      (Native.entry api (software .fp32 plans) p a word call).tensor.shape = logical.shape ∧
      (∀ o, o < p.batches → Host.Result p a real o
        ((Physical.logicalRead api (Native.entry api (software .fp32 plans) p a word call).memory
          (Native.entry api (software .fp32 plans) p a word call).tensor o).toInt)) ∧
      (∀ b, call.caller b → api.byte (Native.entry api (software .fp32 plans) p a word call).memory b = api.byte call.before b) := by
  refine ⟨by simp [Admission.guarded,checked],Admission.checked_abi api p a word call observed checked,?_⟩
  exact whole_fp rest api plans p a word real call
    (Admission.checked_valid api p a word call observed checked) fresh inputs
    (Admission.checked_domain api p a word call observed real checked readback)
    (fun _ pid _ => Admission.checked_seeds api p a word call fresh undef pid) nodes

abbrev exact_returned_words := @Admission.checked_bits
abbrev selected_load_bounds := @Admission.checked_label_read

#axiomsClean checked_returned
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MultinomialSampling.FPEquiv
