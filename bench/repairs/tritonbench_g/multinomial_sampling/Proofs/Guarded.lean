import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Contract

namespace VeriTile.Bench.Repairs.MultinomialSampling.Admission
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MultinomialSampling.SourceValue
noncomputable section

def guarded {α Store Byte : Type} [Inhabited α] (api : Physical.API α Store Byte)
    (P : Primitives α) (p : Parameters) (a : Host.Arguments) (s : State α)
    (call : Native.Preparation api p a s) (o : Observations p) : Option (Native.Returned Store) :=
  if check api p a s call o then some (Native.entry api P p a s call) else none

/-- Independent input recurrence for the exact signed64 returned word. It
uses the canonical Philox and selected FP32 operation trees through `P`. -/
def rawResult {α : Type} [Inhabited α] (P : Primitives α)
    (p : Parameters) (a : Host.Arguments) (s : State α) (r : Nat) : BitVec 64 :=
  BitVec.ofInt 64 (if p.tokens = 1 then Host.inputLabels p a s r 0 else
    VeriTile.Bench.Repairs.MultinomialSampling.SourceValue.Host.value P .fp32
      (Host.params p a) (Host.prepared p a s) r)

theorem returned_word {α Store Byte : Type} [Inhabited α]
    (api : Physical.API α Store Byte) (P : Primitives α) (p : Parameters)
    (a : Host.Arguments) (s : State α) (call : Native.Preparation api p a s)
    (r : Nat) (hr : r < p.batches) :
    Physical.logicalRead api (Native.entry api P p a s call).memory
      (Native.entry api P p a s call).tensor r = rawResult P p a s r := by
  rw [(Native.returned_descriptor api P p a s call).1]
  unfold Physical.logicalRead
  rw [call.outputAddress r hr]
  by_cases ht : p.tokens = 1
  · simpa only [Native.entry,rawResult,ht,↓reduceIte] using call.clone r hr
  · simp only [Native.entry,rawResult,ht,↓reduceIte]
    exact Physical.readback api P (Host.params p a) (Host.prepared p a s) call.output call.memory r hr

#specNonCircular rawResult avoiding [guarded,Host.invoke,Native.entry,Physical.run,FP.Structural.exec]
#axiomsClean returned_word
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MultinomialSampling.Admission
