import bench.repairs.tritonbench_g.lightning_attention.Proofs.ByteHost
namespace VeriTile.Bench.Repairs.LightningAttention.FPEquiv
open VeriTile Triton FP.Structural Host Native Checked
noncomputable section
abbrev forwardReference := Native.forwardReference
abbrev backwardReference := Native.backwardReference

/-- Checked complete native forward with current direct input words, returned
Tensor/metadata and saved context, and every output word and exterior cell. -/
specification lightning_forward_native (I : Instructions) (plan : Plan)
    (c : Invocation) (a : Allocations) (env : LaunchEnvironment)
    (observed : Observations) (admitted : InputContract c.dims observed (forwardTensors a))
    (pointers : PointerBinding observed (forwardTensors a)) (api : ForwardAPI c a env)
    (direct : Direct c a) (current : Nat → Cell Word) (fallback : Memory Word) :
    Spec.Real (ForwardCopies id c a current current ∧ ∃ result,
      nativeForward I plan c a current fallback=some result ∧
      result.output=a.output ∧ result.context.query=c.query.tensor ∧
      result.context.key=c.key.tensor ∧ result.context.value=c.value.tensor ∧
      result.output.shape=valueShape c.dims ∧ result.output.dtype=.fp32 ∧
      result.output.device=c.query.tensor.device ∧
      (∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.E),
        result.physical (a.output.base+packed c.dims c.dims.E b h t f)=
          .mk .real (forwardReference I plan c.dims (logicalWords a current fallback)
            "output" (packed c.dims c.dims.E b h t f))) ∧
      (∀ address,¬ChunkLinearAttn.Physical.Contains (buffer "output" a.output) address →
        result.physical address=current address)) :=
  ⟨direct_reads id c a direct current,
    native_forward I plan c a env (forward_contract c a env observed admitted pointers api) direct current fallback⟩

/-- Backward executes intra then inter on the actual freshly written gradient
words. The cotangent copy is the ordinary retained autograd materialization. -/
specification lightning_backward_native (I : Instructions) (plan : Plan)
    (c : Invocation) (a : Allocations) (env : LaunchEnvironment)
    (observed : Observations) (admitted : InputContract c.dims observed (backwardTensors a))
    (pointers : PointerBinding observed (backwardTensors a)) (api : BackwardAPI c a env)
    (forwardBefore incomingBefore current : Nat → Cell Word) (fallback : Memory Word)
    (direct : Direct c a) (_resolved : c.incoming.lazyNeg=Bool.false)
    (incoming : IncomingCopy id c a incomingBefore current)
    (copyInputs : ∀ t∈backwardProtected c a,∀ off,off<span t →
      current (t.base+off)=incomingBefore (t.base+off))
    (saved : SavedInputs c a forwardBefore current)
    (copyFrame : ∀ address,¬ChunkLinearAttn.Physical.Contains
      ⟨"copy",a.incomingDense.base,span a.incomingDense⟩ address → current address=incomingBefore address) :
    Spec.Real (ForwardCopies id c a forwardBefore current ∧
      IncomingCopy id c a incomingBefore current ∧ ∃ result,
      nativeBackward I plan c a current fallback=some result ∧
      result.gradients=(a.queryGradient,a.keyGradient,a.valueGradient) ∧
      result.gradients.1.shape=queryShape c.dims ∧ result.gradients.2.1.shape=queryShape c.dims ∧
      result.gradients.2.2.shape=valueShape c.dims ∧
      (∀ t∈[result.gradients.1,result.gradients.2.1,result.gradients.2.2],
        t.dtype=.fp32 ∧ t.device=c.query.tensor.device) ∧
      (∀ b∈backwardBuffers a,∀ off,off<b.words →
        result.physical (b.base+off)=.mk .real
          (backwardReference I plan c.dims (logicalWords a current fallback) b.region off)) ∧
      (∀ t∈backwardProtected c a,∀ off,off<span t →
        result.physical (t.base+off)=incomingBefore (t.base+off)) ∧
      (∀ address,¬ChunkLinearAttn.Physical.Contains
          ⟨"copy",a.incomingDense.base,span a.incomingDense⟩ address →
        (∀ b∈backwardBuffers a,¬ChunkLinearAttn.Physical.Contains b address) →
        result.physical address=incomingBefore address)) := by
  have reads:=direct_reads id c a direct forwardBefore
  have later : ForwardCopies id c a forwardBefore current :=
    ⟨fun b h t f=>(saved.query b h t f).trans (reads.query b h t f),
     fun b h t f=>(saved.key b h t f).trans (reads.key b h t f),
     fun b h t f=>(saved.value b h t f).trans (reads.value b h t f)⟩
  obtain ⟨result,he,hg,hq,hk,hv,hm,ho,hinputs,hf⟩:=native_backward I plan c a env
    (backward_contract c a env observed admitted pointers api) current fallback
  refine ⟨later,incoming,result,he,hg,hq,hk,hv,hm,ho,?_,?_⟩
  · intro t ht off ho
    exact (hinputs t ht off ho).trans (copyInputs t ht off ho)
  · intro address hc hg
    exact (hf address hg).trans (copyFrame address hc)

/-- Only local scalar add and a fixed K8 instruction coordinate are deferred.
This transports the complete returned record, including every physical cell. -/
theorem lightning_native_if_local (actual selected : Instructions) (numeric : LocalConformance actual selected)
    (plan : Plan) (c : Invocation) (a : Allocations) (current : Nat → Cell Word) (fallback : Memory Word) :
    nativeForward actual plan c a current fallback=nativeForward selected plan c a current fallback ∧
    nativeBackward actual plan c a current fallback=nativeBackward selected plan c a current fallback := by
  rw [instructions_eq actual selected numeric]
  exact ⟨rfl,rfl⟩

/-- Original inputs are decoded from current little-endian backing bytes.
The same full source execution returns the exact original Tensor/context, all
FP32 words and four-byte stores, with every exterior byte preserved. -/
specification lightning_forward_bytes (I : Instructions) (plan : Plan)
    (c : Invocation) (a : Allocations) (env : LaunchEnvironment)
    (observed : Observations) (admitted : InputContract c.dims observed (forwardTensors a))
    (pointers : PointerBinding observed (forwardTensors a)) (api : ForwardAPI c a env)
    (direct : Direct c a) (current : NativeBytes.Bytes) (fallback : Memory Word) :
    Spec.Real (ForwardCopies id c a (NativeBytes.fromBytes current) (NativeBytes.fromBytes current) ∧
      ∃ result,ByteHost.forward I plan c a current fallback=some result ∧
      nativeForward I plan c a (NativeBytes.fromBytes current) fallback=some result.words ∧
      result.words.output=a.output ∧ result.words.context.query=c.query.tensor ∧
      result.words.context.key=c.key.tensor ∧ result.words.context.value=c.value.tensor ∧
      result.words.output.shape=valueShape c.dims ∧ result.words.output.dtype=.fp32 ∧
      result.words.output.device=c.query.tensor.device ∧
      NativeBytes.fromBytes result.bytes=result.words.physical ∧
      (∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (t : Fin c.dims.N) (f : Fin c.dims.E) (i : Fin 4),
        result.bytes (observed.pointer a.output+4*packed c.dims c.dims.E b h t f+i.val)=
          NativeBytes.encode (forwardReference I plan c.dims
            (logicalWords a (NativeBytes.fromBytes current) fallback) "output" (packed c.dims c.dims.E b h t f)) i) ∧
      (∀ address,¬NativeBytes.Contains (buffer "output" a.output) address → result.bytes address=current address)) := by
  obtain ⟨inputs,wordResult,he,ho,hq,hk,hv,hs,ht,hd,values,frame⟩ :=
    lightning_forward_native I plan c a env observed admitted pointers api direct (NativeBytes.fromBytes current) fallback
  obtain ⟨bytes,hb,decode,byteFrame⟩ := ByteHost.forward_lift I plan c a current fallback wordResult he
  refine ⟨inputs,⟨wordResult,bytes⟩,hb,he,ho,hq,hk,hv,hs,ht,hd,decode,?_,byteFrame⟩
  intro b h t f i
  have result := ByteHost.byte_of_word bytes wordResult.physical decode _ _ (values b h t f) i
  rw [pointers a.output (by simp [forwardTensors])]
  simpa only [Nat.mul_add,Nat.add_assoc] using result

/-- Full autograd backward, including the retained cotangent copy. Its only
copy premises concern input bytes, their protected inputs and copy footprint;
the two kernels and every gradient byte are derived from actual execution. -/
specification lightning_backward_bytes (I : Instructions) (plan : Plan)
    (c : Invocation) (a : Allocations) (env : LaunchEnvironment)
    (observed : Observations) (admitted : InputContract c.dims observed (backwardTensors a))
    (pointers : PointerBinding observed (backwardTensors a)) (api : BackwardAPI c a env)
    (forwardBefore incomingBefore current : NativeBytes.Bytes) (fallback : Memory Word)
    (direct : Direct c a) (resolved : c.incoming.lazyNeg=Bool.false)
    (incoming : ByteHost.IncomingBytes c a incomingBefore current)
    (saved : SavedInputs c a (NativeBytes.fromBytes forwardBefore) (NativeBytes.fromBytes current)) :
    Spec.Real (ForwardCopies id c a (NativeBytes.fromBytes forwardBefore) (NativeBytes.fromBytes current) ∧
      IncomingCopy id c a (NativeBytes.fromBytes incomingBefore) (NativeBytes.fromBytes current) ∧
      ∃ result,ByteHost.backward I plan c a current fallback=some result ∧
      nativeBackward I plan c a (NativeBytes.fromBytes current) fallback=some result.words ∧
      result.words.gradients=(a.queryGradient,a.keyGradient,a.valueGradient) ∧
      result.words.gradients.1.shape=queryShape c.dims ∧ result.words.gradients.2.1.shape=queryShape c.dims ∧
      result.words.gradients.2.2.shape=valueShape c.dims ∧
      (∀ t∈[result.words.gradients.1,result.words.gradients.2.1,result.words.gradients.2.2],
        t.dtype=.fp32 ∧ t.device=c.query.tensor.device) ∧
      NativeBytes.fromBytes result.bytes=result.words.physical ∧
      (∀ b∈backwardBuffers a,∀ off,off<b.words → ∀ i : Fin 4,
        result.bytes (4*(b.base+off)+i.val)=NativeBytes.encode
          (backwardReference I plan c.dims (logicalWords a (NativeBytes.fromBytes current) fallback) b.region off) i) ∧
      (∀ t∈backwardProtected c a,∀ off,off<span t → ∀ i : Fin 4,
        result.bytes (4*(t.base+off)+i.val)=incomingBefore (4*(t.base+off)+i.val)) ∧
      (∀ address,¬NativeBytes.Contains ⟨"copy",a.incomingDense.base,span a.incomingDense⟩ address →
        (∀ b∈backwardBuffers a,¬NativeBytes.Contains b address) → result.bytes address=incomingBefore address)) := by
  have copied := ByteHost.incoming_words c a incomingBefore current incoming resolved
  obtain ⟨inputs,copy,wordResult,he,hg,hq,hk,hv,hm,values,inputFrame,frame⟩ :=
    lightning_backward_native I plan c a env observed admitted pointers api
      (NativeBytes.fromBytes forwardBefore) (NativeBytes.fromBytes incomingBefore) (NativeBytes.fromBytes current)
      fallback direct resolved copied (ByteHost.copy_inputs_words c a incomingBefore current incoming) saved
      (ByteHost.copy_frame_words c a incomingBefore current incoming)
  obtain ⟨bytes,hb,decode,byteFrame⟩ := ByteHost.backward_lift I plan c a current fallback wordResult he
  refine ⟨inputs,copy,⟨wordResult,bytes⟩,hb,he,hg,hq,hk,hv,hm,decode,?_,?_,?_⟩
  · intro b hb off ho i
    exact ByteHost.byte_of_word bytes wordResult.physical decode _ _ (values b hb off ho) i
  · intro t ht off ho i
    apply ByteHost.byte_frame_of_cells
    rw [decode]
    exact inputFrame t ht off ho
  · intro address copyOutside outputOutside
    exact (byteFrame address outputOutside).trans (incoming.frame address copyOutside)

/-- Local instruction transport also preserves the complete byte-backed return. -/
theorem lightning_bytes_if_local (actual selected : Instructions) (numeric : LocalConformance actual selected)
    (plan : Plan) (c : Invocation) (a : Allocations) (current : NativeBytes.Bytes) (fallback : Memory Word) :
    ByteHost.forward actual plan c a current fallback=ByteHost.forward selected plan c a current fallback ∧
    ByteHost.backward actual plan c a current fallback=ByteHost.backward selected plan c a current fallback := by
  rw [instructions_eq actual selected numeric]
  exact ⟨rfl,rfl⟩

#axiomsClean lightning_forward_bytes
#axiomsClean lightning_backward_bytes
#axiomsClean lightning_bytes_if_local
#specNonCircular forwardReference avoiding [nativeForward,nativeBackward,FP.Structural.exec]
#specNonCircular backwardReference avoiding [nativeForward,nativeBackward,FP.Structural.exec]
#axiomsClean lightning_forward_native
#axiomsClean lightning_backward_native
#axiomsClean lightning_native_if_local
end
end VeriTile.Bench.Repairs.LightningAttention.FPEquiv
