import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Model
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Numerics

namespace VeriTile.Bench.Repairs.MultinomialSampling.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MultinomialSampling.SourceValue
noncomputable section
set_option maxHeartbeats 3000000

/-- Full software FP32 source and actual native store trace meet the independent
categorical relation under the explicitly local numerical conditions. The
canonical Philox sample, scan/max schedules, discrete reductions, selection,
coverage and memory are proved rather than deferred. -/
theorem whole_fp (rest : Algebra Word) (api : Physical.API Word Store Byte) (plans : Plans)
    (p : Parameters) (a : Host.Arguments) (word : State Word) (real : State ℝ)
    (call : Native.Preparation api p a word)
    (valid : Host.Valid p a) (fresh : Host.Fresh p a) (inputs : Numerics.Inputs p word real)
    (domain : ∀ o, o < p.batches → Spec.Domain p.tokens (Host.inputWeights p real o))
    (seeds : p.tokens ≠ 1 → ∀ k, k < VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.grid (Host.params p a) →
      SeedDomain (Host.params p a) (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.programState (Host.params p a) (Host.prepared p a word) (Host.prepared p a word) k))
    (nodes : p.tokens ≠ 1 → Numerics.LaunchExact plans p a word) :
    ∃ logical, Host.invoke rest (software .fp32 plans) p a word = some logical ∧
      logical.shape = (if p.tokens=1 then [p.batches,1] else [p.batches]) ∧
      (Native.entry api (software .fp32 plans) p a word call).tensor = call.output ∧
      (Native.entry api (software .fp32 plans) p a word call).tensor.shape = logical.shape ∧
      (∀ o, o < p.batches → Host.Result p a real o
        ((Physical.logicalRead api (Native.entry api (software .fp32 plans) p a word call).memory
          (Native.entry api (software .fp32 plans) p a word call).tensor o).toInt)) ∧
      (∀ b, call.caller b → api.byte (Native.entry api (software .fp32 plans) p a word call).memory b = api.byte call.before b) := by
  obtain ⟨logical,he,_,hshape⟩ := Host.wrapper_correct rest (software .fp32 plans) p a word valid fresh.1 seeds
  have descriptor := Native.returned_descriptor api (software .fp32 plans) p a word call
  refine ⟨logical,he,hshape,descriptor.1,descriptor.2.trans hshape.symm,?_,
    Native.caller_frame api (software .fp32 plans) p a word call⟩
  intro o ho
  rw [descriptor.1]
  unfold Physical.logicalRead
  rw [call.outputAddress o ho]
  by_cases ht : p.tokens=1
  · simp only [Native.entry,ht,↓reduceIte]
    rw [call.clone o ho,call.labelsRoundtrip o 0 ho (by omega),Numerics.input_labels_eq p a word real inputs]
    unfold Host.Result
    rw [ht]
    apply Spec.singleton
    simpa only [ht] using domain o ho
  · simp only [Native.entry,ht,↓reduceIte]
    rw [Physical.readback api (software .fp32 plans) (Host.params p a) (Host.prepared p a word) call.output call.memory o ho]
    rw [Numerics.value_eq plans p a word real inputs fresh (nodes ht) o ho]
    obtain ⟨j,hj,hv⟩ := Host.value_result p a real fresh o ho (domain o ho)
    unfold Host.Result Spec.Result
    refine ⟨j,hj,?_⟩
    rw [hv,← Numerics.input_labels_eq p a word real inputs o j,call.labelsRoundtrip o j ho hj.1]

/-- Unconditional exact bit-profile execution of the complete repaired host;
it is separate from the Real categorical refinement's local numeric ports. -/
abbrev whole_bits := @Host.wrapper_correct

#axiomsClean whole_fp
#axiomsClean whole_bits
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MultinomialSampling.FPEquiv
