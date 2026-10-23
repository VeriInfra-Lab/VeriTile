import bench.repairs.tritonbench_g.attention_llama.Proofs.NativeBinding

namespace VeriTile.Bench.Repairs.AttentionLlama.Checked
open VeriTile Triton
open LoopInvariant HostGrid HostMemory Correctness FPEquivalence
set_option maxHeartbeats 1000000

/-- Checked tensor return and both physical frames for the preserved native
primitive graph. Local numerical correspondence remains explicitly deferred;
no exact-to-Real premise is used to establish addresses or safety. -/
theorem attention_native_graph {α β : Type} (B : Nat) (c : Context) (o : OutputLayout B c)
    (fp8 : Bool) (q k v : TensorMetadata) (a : ScalarMetadata) (p : Pointers B c o)
    (hinput : inputChecks B c fp8 q k v a=Bool.true) (hl : launchChecks B c o=Bool.true)
    (hp : pointerChecks B c o fp8 p=Bool.true)
    (A : NativeGraph.Ops α) (R : NativeGraph.Ops β) (f : α→β)
    (input : FPEquivalence.TensorInputs α B c.NH c.H c.D) (scale : α)
    (hlocal : numericalDomain A R f input.at scale fp8 c.causal c.N c.SP)
    (memory : Nat→α) (referenceMemory : Nat→β) :
    let address := outputAddress (B:=B) c o.base o.soz o.soh o.som o.son
    (∀ i, f ((returnTensor address
      (storeAll address (output A input.at scale fp8 c.causal c.N c.SP) memory
        (schedule B c.NH c.H c.D))).read i)=
      (returnTensor address
        (storeAll address (output R (fun i=>(input.at i).map f) (f scale) fp8 c.causal c.N c.SP)
          referenceMemory (schedule B c.NH c.H c.D))).read i) ∧
    (∀ location, (∀ i, location≠address i) →
      storeAll address (output A input.at scale fp8 c.causal c.N c.SP) memory
        (schedule B c.NH c.H c.D) location=memory location) ∧
    (∀ location, (∀ i, location≠address i) →
      storeAll address (output R (fun i=>(input.at i).map f) (f scale) fp8 c.causal c.N c.SP)
        referenceMemory (schedule B c.NH c.H c.D) location=referenceMemory location) ∧
    (∀ i, 2*address i<2^64) ∧ supportedHead c.D ∧
    (∀ n : Nat, n<envelope B c o → (BitVec.ofInt 32 (n : Int)).toInt=(n : Int)) := by
  have hm:=input_metadata B c fp8 q k v a hinput
  simp only [metadataChecks,decide_eq_true_eq] at hm
  obtain ⟨hr,hf,hg⟩:=native_graph_transport A R f input.at scale fp8 c.causal c.N c.SP hlocal
    (outputAddress c o.base o.soz o.soh o.som o.son) o.injective memory referenceMemory
  exact ⟨hr,hf,hg,returned_address B c o fp8 p hp,hm.1,formed_signed32 B c o hl⟩

#axiomsClean attention_native_graph

/-- Complete checked native caller return. Input and scalar words are read
from the actual caller objects/ABI; native output half words are stored as two
bytes, read through the actual returned FP16 Tensor descriptor, and framed on
finite physical memory. Only local primitive numerical transport is deferred. The source-to-this-
recorded-SM80-graph mapping remains the ordinary pinned Triton compiler/ISA
contract documented by the retained profile audit. This theorem adds no proof
of compiler internals and assumes no native-to-Real equality. -/
theorem attention_native {α β : Type} (B : Nat) (c : Context) (o : OutputLayout B c)
    (fp8 : Bool) (q k v : TensorMetadata) (a : ScalarMetadata) (p : Pointers B c o)
    (hinput : inputChecks B c fp8 q k v a=Bool.true) (hl : launchChecks B c o=Bool.true)
    (hp : pointerChecks B c o fp8 p=Bool.true)
    (A : NativeGraph.Ops α) (R : NativeGraph.Ops β) (f : α→β) (codec : HalfCodec A)
    (objects : InputObjects B c o fp8 p) (scalar : ScalarWord α)
    (scalarBinding : scalar.project scalar.argument=c.scale)
    (inputDevices : objects.query.device=q.device ∧ objects.key.device=k.device ∧ objects.value.device=v.device)
    (allocated : TensorDescriptor) (allocation : OutputBinding B c o p q allocated)
    (fresh : FreshInputs objects allocated) (caller : ByteMemory)
    (hlocal : numericalDomain A R f (actualInputs B c o A codec fp8 p objects caller).at
      scalar.value fp8 c.causal c.N c.SP) :
    let input := actualInputs B c o A codec fp8 p objects caller
    let word := output A input.at scalar.value fp8 c.causal c.N c.SP
    let result := finishBytes B c o allocated (fun i=>codec.encode (word i)) caller
    result.tensor=allocated ∧ result.tensor.dtype=DType.half ∧
    result.tensor.shape=[B,c.NH,c.H,c.D] ∧ result.tensor.strides=[o.soz,o.soh,o.som,o.son] ∧
    result.tensor.device=q.device ∧ result.tensor.dataPointer=p.base .output ∧
    (∀ i, result.read i=codec.encode (word i)) ∧
    (∀ i, f (codec.decode (result.read i))=
      output R (fun i=>(input.at i).map f) (f scalar.value) fp8 c.causal c.N c.SP i) ∧
    (∀ address : Fin (2^64), (∀ i, address.val≠outputByte B c o i) → result.memory address=caller address) ∧
    (∀ t∈[objects.query,objects.key,objects.value], ∀ address : Fin (2^64),
      t.allocationBase≤address.val → address.val<t.allocationBase+t.allocationBytes →
      result.memory address=caller address) ∧
    input=observedInputs B c o A codec p caller ∧ scalar.word=scalar.marshal scalar.argument ∧
    scalar.project scalar.argument=c.scale ∧
    (objects.query.device=q.device ∧ objects.key.device=k.device ∧ objects.value.device=v.device) ∧
    supportedHead c.D ∧
    (∀ n : Nat, n<envelope B c o → (BitVec.ofInt 32 (n : Int)).toInt=(n : Int)) := by
  have hm:=input_metadata B c fp8 q k v a hinput
  simp only [metadataChecks,decide_eq_true_eq] at hm
  refine ⟨rfl,allocation.dtype,allocation.shape,allocation.strides,allocation.device,allocation.pointer,
    finish_read B c o allocated _ caller fp8 p hp q allocation,?_,
    finish_frame B c o allocated _ caller,
    finish_inputs_frame B c o allocated _ caller fp8 p hp q allocation objects fresh,
    actual_inputs_bind B c o A codec fp8 p objects caller,scalar.marshalled,scalarBinding,inputDevices,hm.1,formed_signed32 B c o hl⟩
  · intro i
    rw [finish_read B c o allocated _ caller fp8 p hp q allocation i]
    have hc : codec.decode (codec.encode (output A
        (actualInputs B c o A codec fp8 p objects caller).at scalar.value fp8 c.causal c.N c.SP i))=
        output A (actualInputs B c o A codec fp8 p objects caller).at scalar.value fp8 c.causal c.N c.SP i :=
      codec.halfRoundtrip _
    rw [hc]
    exact NativeLocal.output_transport A R f _ scalar.value fp8 c.causal c.N c.SP
      (i.query.val/64) (i.query.val%64) i.channel (hlocal i)

#axiomsClean attention_native
end VeriTile.Bench.Repairs.AttentionLlama.Checked
