import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.NativeBacking

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Backing
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionLlama HostGrid
open Source
noncomputable section
set_option maxHeartbeats 1000000

theorem layout_address (t : TensorObject) (p : Pointer)
    (sz sh sn sd : Nat) (dtype : Checked.DType)
    (h : TensorLayout t p B NH H D [sz,sh,sn,sd] dtype) (i : Index B NH H D) :
    tensorAddress t i=p.address (i.batch.val*sz+i.head.val*sh+i.query.val*sn+i.channel.val*sd) := by
  simp only [tensorAddress,elementOffset,h.pointer,h.stride,h.width,Pointer.address,
    List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some]

/-- The logical caller inputs are observations of the actual Tensor objects
in the very same initial heap later updated by `nativePost`. E5M2 byte payloads
are decoded only after reading the key/value object's pointer and strides. -/
theorem caller_observations (c : Config) (p : RegionName→Pointer) (o : Objects B c p)
    (before : Heap (Cell (Term α))) (seed : State (Term α))
    (x : Tensor.Inputs α B c.NH c.H c.D)
    (binding : Tensor.Binding c (sourceState p before seed) x) :
    (∀ i : Index B c.NH c.H c.D,
      (tensorCell before o.query i).read .fp16=.leaf (x.query i.batch i.head i.query i.channel)) ∧
    (c.fp8=Bool.false →
      (∀ i : Index B c.NH c.N c.D, (tensorCell before o.key i).read .fp16=
        .leaf (x.halfKey i.batch i.head i.query.val i.channel)) ∧
      (∀ i : Index B c.NH c.N c.D, (tensorCell before o.value i).read .fp16=
        .leaf (x.halfValue i.batch i.head i.query.val i.channel))) ∧
    (c.fp8=Bool.true →
      (∀ i : Index B c.NH c.N c.D, UInt8.ofFin (Float8E5Bits.ofInt ((tensorCell before o.key i).read .int))=
        x.byteKey i.batch i.head i.query.val i.channel) ∧
      (∀ i : Index B c.NH c.N c.D, UInt8.ofFin (Float8E5Bits.ofInt ((tensorCell before o.value i).read .int))=
        x.byteValue i.batch i.head i.query.val i.channel)) := by
  refine ⟨?_,?_,?_⟩
  · intro i
    rw [tensorCell,layout_address _ _ _ _ _ _ _ o.queryLayout]
    exact binding.query i.batch i.head i.query i.channel
  · intro hf
    constructor
    · intro i
      rw [tensorCell,layout_address _ _ _ _ _ _ _ o.keyLayout]
      exact binding.halfKey hf i.batch i.head i.query.val i.channel i.query.isLt
    · intro i
      rw [tensorCell,layout_address _ _ _ _ _ _ _ o.valueLayout]
      exact binding.halfValue hf i.batch i.head i.query.val i.channel i.query.isLt
  · intro hf
    constructor
    · intro i
      rw [tensorCell,layout_address _ _ _ _ _ _ _ o.keyLayout]
      exact binding.byteKey hf i.batch i.head i.query.val i.channel i.query.isLt
    · intro i
      rw [tensorCell,layout_address _ _ _ _ _ _ _ o.valueLayout]
      exact binding.byteValue hf i.batch i.head i.query.val i.channel i.query.isLt

#axiomsClean layout_address
#axiomsClean caller_observations
#specNonCircular sourceState avoiding [nativePost,Tensor.expected,Target.output]
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Backing
