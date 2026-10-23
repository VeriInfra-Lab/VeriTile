import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Model

namespace VeriTile.Bench.Repairs.MultinomialSampling.Correct
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MultinomialSampling.SourceValue
noncomputable section

abbrev whole_real := @Host.whole_real

/-- Same-call public host execution, actual returned descriptor, independent
categorical result, and every preexisting caller byte. Native compiler/runtime
binding is restricted to the typed local API and source finite-width ABI. -/
theorem whole_native (rest : Algebra ℝ) (api : Physical.API ℝ Store Byte)
    (p : Parameters) (a : Host.Arguments) (s : State ℝ) (call : Native.Preparation api p a s)
    (valid : Host.Valid p a) (fresh : Host.Fresh p a)
    (domain : ∀ o, o < p.batches → Spec.Domain p.tokens (Host.inputWeights p s o))
    (seeds : p.tokens ≠ 1 → ∀ k, k < VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.grid (Host.params p a) →
      SeedDomain (Host.params p a) (VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.programState (Host.params p a) (Host.prepared p a s) (Host.prepared p a s) k))
    :
    ∃ logical, Host.invoke rest Real.primitives p a s = some logical ∧
      logical.shape = (if p.tokens=1 then [p.batches,1] else [p.batches]) ∧
      (Native.entry api Real.primitives p a s call).tensor = call.output ∧
      (Native.entry api Real.primitives p a s call).tensor.shape = logical.shape ∧
      (∀ o, o < p.batches → Host.Result p a s o
        ((Physical.logicalRead api (Native.entry api Real.primitives p a s call).memory
          (Native.entry api Real.primitives p a s call).tensor o).toInt)) ∧
      (∀ b, call.caller b → api.byte (Native.entry api Real.primitives p a s call).memory b = api.byte call.before b) := by
  obtain ⟨logical,he,hshape,_,_⟩ := Host.whole_real rest p a s valid fresh domain seeds
  obtain ⟨hd,hs,hv,hf⟩ := Native.whole_native api p a s call fresh domain
  exact ⟨logical,he,hshape,hd,hs.trans hshape.symm,hv,hf⟩

#axiomsClean whole_real
#axiomsClean whole_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MultinomialSampling.Correct
