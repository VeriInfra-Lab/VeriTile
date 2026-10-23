import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Guarded
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPSoftware

namespace VeriTile.Bench.Repairs.MultinomialSampling.Admission
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MultinomialSampling.SourceValue
noncomputable section

theorem checked_bits (rest : Algebra Word) (api : Physical.API Word Store Byte) (plans : Plans)
    (p : Parameters) (a : Host.Arguments) (s : State Word)
    (call : Native.Preparation api p a s) (o : Observations p)
    (checked : check api p a s call o = Bool.true)
    (fresh : Host.Fresh p a) (undef : TypedUndefined p s) :
    ∃ logical returned,
      Host.invoke rest (software .fp32 plans) p a s = some logical ∧
      guarded api (software .fp32 plans) p a s call o = some returned ∧
      returned.tensor = call.output ∧ returned.tensor.shape =
        (if p.tokens = 1 then [p.batches,1] else [p.batches]) ∧
      (∀ r, r < p.batches → Physical.logicalRead api returned.memory returned.tensor r =
        rawResult (software .fp32 plans) p a s r) ∧
      (∀ b, call.caller b → api.byte returned.memory b = api.byte call.before b) ∧
      (p.tokens ≠ 1 → IndexABI api p a s call) := by
  obtain ⟨logical,hl,_,_⟩ := Host.wrapper_correct rest (software .fp32 plans) p a s
    (checked_valid api p a s call o checked) fresh.1
    (fun _ pid _ => checked_seeds api p a s call fresh undef pid)
  refine ⟨logical,Native.entry api (software .fp32 plans) p a s call,hl,?_,
    (Native.returned_descriptor api (software .fp32 plans) p a s call).1,
    (Native.returned_descriptor api (software .fp32 plans) p a s call).2,
    returned_word api (software .fp32 plans) p a s call,
    Native.caller_frame api (software .fp32 plans) p a s call,
    checked_abi api p a s call o checked⟩
  simp [guarded,checked]

#axiomsClean checked_bits
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MultinomialSampling.Admission
