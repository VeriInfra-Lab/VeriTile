import bench.repairs.tritonbench_g.triton_attention.Proofs.Composition

namespace VeriTile.Bench.Repairs.TritonAttention.Whole001.Checked
open Native Memory HostSeparate
noncomputable section
open scoped Classical
set_option maxHeartbeats 3500000

/-- Canonical FP32 forward/backward composition. The forward endpoint consumes
only Q/K/V, and the subsequent backward endpoint consumes the actual incoming
cotangent. All four returned tensors are read from final physical bytes; the
reference depends only on initial Q/K/V and that later cotangent. -/
theorem whole_native (g : Geometry) (hs : shapeChecks g=Bool.true) (device forwardPrior backwardPrior : Nat)
    (objects : Objects (checkedShape g hs) device) (incoming : Tensor)
    (actual reference : Instructions) (owners : Owners) (before : ByteMemory) (scale : ScalarABI)
    (forwardChecksPassed : forwardChecks g objects.forwardObjects.complete.tensor scale=Bool.true)
    (backwardChecksPassed : backwardChecks g device objects.tensor incoming=Bool.true)
    (forwardAllocation : ForwardAllocation objects.forwardObjects.complete.tensor)
    (backwardAllocation : BackwardAllocation objects.tensor incoming)
    (numeric : LocalConformance actual reference) :
    ∃ f b,
      forwardCall g hs device forwardPrior objects.forwardObjects actual owners before scale=some f ∧
      backwardCall g hs device backwardPrior objects incoming actual owners f.memory scale=some b ∧
      f.output=objects.tensor .o ∧ f.saved=forwardLive.map objects.tensor ∧
      b.dq=objects.tensor .dq ∧ b.dk=objects.tensor .dk ∧ b.dv=objects.tensor .dv ∧ b.scaleGradient=none ∧
      f.rawScale=scale.raw ∧ b.rawScale=scale.raw ∧
      f.launchDevice=device ∧ b.launchDevice=device ∧
      f.restoredDevice=forwardPrior ∧ b.restoredDevice=backwardPrior ∧
      WholeValues (checkedShape g hs) objects.forwardObjects.complete.tensor objects.tensor incoming
        reference owners before f.memory b.memory scale.word ∧
      (∀ i : ByteIndex (checkedShape g hs) backwardWrites,
        extendMemory b.memory (byteAddress _ objects.tensor backwardWrites i)=
          encode (backwardWords (checkedShape g hs) objects.tensor incoming actual owners f.memory scale.word i.1.val i.2.1.val) i.2.2) ∧
      (∀ location : Fin (2^64),
        (∀ i : ByteIndex (checkedShape g hs) forwardWrites,
          location.val≠byteAddress _ objects.forwardObjects.complete.tensor forwardWrites i) →
        (∀ i : ByteIndex (checkedShape g hs) backwardWrites,location.val≠byteAddress _ objects.tensor backwardWrites i) →
        (∀ i : ByteIndex (checkedShape g hs) [.g],location.val≠byteAddress _ objects.tensor [.g] i) →
        b.memory location=before location) ∧
      (∀ input∈[Buffer.q,.k,.v],∀ location : Fin (2^64),
        (objects.tensor input).storageBase≤location.val →
        location.val<(objects.tensor input).storageBase+(objects.tensor input).storageBytes →
        b.memory location=before location) ∧
      (∀ location : Fin (2^64),incoming.storageBase≤location.val →
        location.val < incoming.storageBase+incoming.storageBytes → b.memory location=f.memory location) := by
  obtain ⟨f,hf,fo,fs,fr,fd,fp,fv,ff,fi⟩:=forward_native g hs device forwardPrior objects.forwardObjects
    actual reference owners before scale forwardChecksPassed forwardAllocation numeric
  have fEq : f=forwardReturn (checkedShape g hs) device forwardPrior objects.forwardObjects.complete.tensor
      actual owners before scale := by
    simpa only [forwardCall,forwardChecksPassed,↓reduceIte,Option.some.injEq] using hf.symm
  subst f
  obtain ⟨b,hb,bq,bk,bv,bnone,br,bd,bp,bvalues,bbytes,bframe,binputs⟩:=
    backward_native g hs device backwardPrior objects incoming actual reference owners
      (forwardBytes (checkedShape g hs) objects.forwardObjects.complete.tensor actual owners before scale.word)
      scale backwardChecksPassed backwardAllocation numeric
  have hp:=forward_pointers g objects.forwardObjects.complete.tensor scale forwardChecksPassed
  have hpb:=backward_pointers g device objects.tensor incoming backwardChecksPassed
  have output : ∀ i : Coord (checkedShape g hs),
      logicalRead (checkedShape g hs) (objects.forwardObjects.complete.tensor .o) b.memory i=
      logicalRead (checkedShape g hs) (objects.forwardObjects.complete.tensor .o)
        (forwardBytes (checkedShape g hs) objects.forwardObjects.complete.tensor actual owners before scale.word) i := by
    intro i
    rw [complete_live objects.forwardObjects .o (by simp [forwardLive])]
    change logicalRead _ (objects.tensor .o) _ _=logicalRead _ (objects.tensor .o) _ _
    rw [logical_read _ device objects _ .o (by decide) (by decide),
      logical_read _ device objects _ .o (by decide) (by decide)]
    exact read_preserved g (objects.tensor .o) (hpb .o (by simp [backwardLive,forwardLive]))
      (objects.width .o (by decide)) _ b.memory
      (binputs (objects.tensor .o) (by simp [forwardLive])) (offset _ i)
      (by rw [objects.count];exact offset_lt _ i)
  refine ⟨forwardReturn _ device forwardPrior objects.forwardObjects.complete.tensor actual owners before scale,b,
    hf,hb,fo,fs,bq,bk,bv,bnone,fr,br,fd,bd,fp,bp,
    composed_values g hs device objects incoming actual reference owners before b.memory scale.word hp
      forwardAllocation numeric output fv bvalues,bbytes,?_,?_,?_⟩
  · intro location fout bout copyout
    exact (bframe location bout copyout).trans (ff location fout)
  · intro input hi location hlo hhi
    have hlive : input∈forwardLive := by
      simp only [List.mem_cons,List.not_mem_nil,or_false] at hi
      rcases hi with rfl|rfl|rfl <;> simp [forwardLive]
    have hmem : objects.tensor input∈incoming::forwardLive.map objects.tensor := by
      simp only [List.mem_cons,List.mem_map]
      exact Or.inr ⟨input,hlive,rfl⟩
    exact (binputs (objects.tensor input) hmem location hlo hhi).trans (fi input hi location hlo hhi)
  · intro location hlo hhi
    exact binputs incoming (by simp) location hlo hhi

#axiomsClean whole_native
end
end VeriTile.Bench.Repairs.TritonAttention.Whole001.Checked
