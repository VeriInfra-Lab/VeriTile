import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.NativeExecution
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.TensorBridge

namespace VeriTile.Bench.Repairs.MixedSparseAttention.Checked.FP
open VeriTile Triton
open VeriTile.Bench.Repairs.MixedSparseAttention.Whole002 Source Whole
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section

/-- Checked FP call through the complete typed source. The only numerical
premises are scalar and fixed-m16n8k16 instruction transports and equality.
Initial raw words, signed metadata, scale ABI word, narrowing/widening and
owner reduction order remain in the executed native graph. -/
theorem checked_call (api : API (BitVec 32) Store (BitVec 8)) (codec : Raw.WordCodec api)
    (actual standard : Native.CompareISA (BitVec 32))
    (laws : Native.ComparisonTransport id actual standard) (c : Config) (layout : Reify.Layout c)
    (request : Request) (m : Store) (launch : Bool)
    (accepted : validate c request (NativeExecution.observed c (Raw.inputs api 0 c request m))=.ok launch)
    (order : List (Raw.Coord c)) (complete : ∀ q,q∈order) :
    ∃ after,Raw.dispatch api 0 c request (NativeExecution.observed c (Raw.inputs api 0 c request m))
      (NativeExecution.source (Native.Runtime.mk request.scaleWord) actual c layout) order m=.ok (request.tensor .output,after) ∧
      ((launch=Bool.false ∧ after=m) ∨ (launch=Bool.true ∧
        (∀ q : Raw.Coord c,api.read after (Raw.address request q)=
          NativeExecution.target (Native.Runtime.mk request.scaleWord) standard c layout (Raw.inputs api 0 c request m) q) ∧
        (∀ (q : Raw.Coord c) (j : Fin 4),api.byte after (Raw.address request q+j.val)=
          Raw.wordByte (NativeExecution.target (Native.Runtime.mk request.scaleWord) standard c layout (Raw.inputs api 0 c request m) q) j) ∧
        (∀ (q : Raw.Coord c) (j : Fin 4),Raw.address request q+j.val<2^64 ∧
          (BitVec.ofNat 64 (Raw.address request q+j.val)).toNat=Raw.address request q+j.val) ∧
        (∀ i,Raw.Outside request order i → api.byte after i=api.byte m i) ∧
        Raw.inputs api 0 c request after=Raw.inputs api 0 c request m ∧
        NativeExecution.observed c (Raw.inputs api 0 c request after)=
          NativeExecution.observed c (Raw.inputs api 0 c request m))) := by
  cases launch
  · exact ⟨m,Raw.dispatch_empty _ _ _ _ _ _ _ _ accepted,Or.inl ⟨rfl,rfl⟩⟩
  · have live:=(validate_launch.mp accepted).2
    have words (q : Raw.Coord c) :
        api.read (Raw.execute api 0 c request (NativeExecution.source (Native.Runtime.mk request.scaleWord) actual c layout) order m)
          (Raw.address request q)=NativeExecution.target (Native.Runtime.mk request.scaleWord) standard c layout (Raw.inputs api 0 c request m) q := by
      rw [Raw.readback api 0 live _ order m q (complete q),
        NativeExecution.source_target (Native.Runtime.mk request.scaleWord) actual standard laws c layout request _ live]
    have stable:=Raw.inputs_execute api 0 live (NativeExecution.source (Native.Runtime.mk request.scaleWord) actual c layout) order m
    refine ⟨_,Raw.dispatch_launch _ _ _ _ _ _ _ _ accepted,Or.inr ⟨rfl,words,?_,
      Raw.output_word live,Raw.byte_frame api 0 live _ order m,stable,?_⟩⟩
    · intro q j;rw [codec.bytes,words q]
    · rw [stable]

#axiomsClean checked_call
#auditStmt checked_call

/-- The same checked call through the actual returned Tensor's ordinary
shared metadata/storage binding and logical FP32 API interpretation. -/
theorem checked_tensor_call (api : API (BitVec 32) Store (BitVec 8)) (codec : Raw.WordCodec api)
    (actual standard : Native.CompareISA (BitVec 32))
    (laws : Native.ComparisonTransport id actual standard) (c : Config) (layout : Reify.Layout c)
    (request : Request) (m : Store) (launch : Bool)
    (accepted : validate c request (NativeExecution.observed c (Raw.inputs api 0 c request m))=.ok launch)
    (order : List (Raw.Coord c)) (complete : ∀ q,q∈order)
    (tensor : VeriTile.Torch.FP32.Tensor) (binding : TensorBridge.Observation (request.tensor .output) tensor) :
    ∃ after,Raw.dispatch api 0 c request (NativeExecution.observed c (Raw.inputs api 0 c request m))
      (NativeExecution.source (Native.Runtime.mk request.scaleWord) actual c layout) order m=
        .ok (request.tensor .output,after) ∧
      (request.tensor .output).object=tensor.object ∧ tensor.metadata.shape=[c.Z,c.H,c.T,c.BD] ∧
      ((launch=Bool.false ∧ after=m) ∨ (launch=Bool.true ∧
        (∀ q : Raw.Coord c,TensorBridge.logicalWord api after tensor binding (Raw.offset q)=
          NativeExecution.target (Native.Runtime.mk request.scaleWord) standard c layout (Raw.inputs api 0 c request m) q) ∧
        (∀ i,Raw.Outside request order i → api.byte after i=api.byte m i))) := by
  obtain ⟨after,dispatch,post⟩:=checked_call api codec actual standard laws c layout request m launch accepted order complete
  cases launch
  · have header:=(validate_empty.mp accepted).schema.header .output
    refine ⟨after,dispatch,binding.object.symm,binding.shape.trans header.1,?_⟩
    rcases post with empty|live
    · exact Or.inl empty
    · cases live.1
  · have live:=(validate_launch.mp accepted).2
    refine ⟨after,dispatch,binding.object.symm,binding.shape.trans (live.schema.header .output).1,?_⟩
    rcases post with empty|post
    · cases empty.1
    · refine Or.inr ⟨rfl,?_,post.2.2.2.2.1⟩
      intro q
      rw [TensorBridge.logical_read api after live tensor binding q]
      exact post.2.1 q

#axiomsClean checked_tensor_call
#auditStmt checked_tensor_call
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Checked.FP
