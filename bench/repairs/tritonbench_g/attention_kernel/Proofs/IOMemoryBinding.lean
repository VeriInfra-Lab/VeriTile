import bench.repairs.tritonbench_g.attention_kernel.Proofs.IOFormulaBinding

namespace VeriTile.Bench.Repairs.AttentionKernel.IOMemoryBinding
open VeriTile Triton FP.Structural Whole001 Source Whole Machine Whole002
open IOFormulaBinding
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false
variable {α β : Type} [Inhabited α] [Inhabited β]

def encodedViews (encode : α → β) (p : Parameters) (vals : ChannelValues α (IO.inputs p))
    (b : Fin p.batches) (h : Fin p.heads) : Target.Views β :=
  let x:=Target.views p vals b h
  ⟨fun r d=>encode (x.q r d),fun r d=>encode (x.k r d),
   fun r d=>encode (x.v r d),fun r c=>encode (x.bias r c)⟩

theorem float_map (encode : α → β) (storage : TypedSource.Storage)
    (x : Value α storage.tile) :
    toFloat storage.float (mapValue encode storage.tile x)=encode (toFloat storage.float x) := by
  cases storage <;> rfl

theorem reads (M : Algebra β) (dot : Dot β) (p : Parameters) (work : List Program)
    (encode : α → β) (observe : TileIndex (IO.TensorShape p) → β → α)
    (vals : ChannelValues α (IO.inputs p)) (s : State β)
    (binding : (IO.attentionIO M dot p work encode observe).InputBinding vals s)
    (b : Fin p.batches) (h : Fin p.heads) :
    Reads p (readState p s) b.val h.val (encodedViews encode p vals b h) := by
  constructor
  · intro r hr d hd
    have ht:=congrArg (toFloat p.storage.float)
      (binding (0 : Fin 4) (b,h,⟨r,hr⟩,⟨d,hd⟩,()) trivial)
    change readState p s p.Q (offset p.q b.val h.val r d)=
      toFloat p.storage.float (mapValue encode p.storage.tile (vals (0 : Fin 4) (b,h,⟨r,hr⟩,⟨d,hd⟩,()))) at ht
    rw [float_map] at ht
    simpa [IO.attentionIO,IO.inputs,IO.address,IO.queryValues,encodedViews,Target.views,Target.get,
      readState,float_map,hr,hd] using ht
  · intro r hr d hd
    have ht:=congrArg (toFloat p.storage.float)
      (binding (1 : Fin 4) (b,h,⟨r,hr⟩,⟨d,hd⟩,()) trivial)
    change readState p s p.K (offset p.k b.val h.val r d)=
      toFloat p.storage.float (mapValue encode p.storage.tile (vals (1 : Fin 4) (b,h,⟨r,hr⟩,⟨d,hd⟩,()))) at ht
    rw [float_map] at ht
    simpa [IO.attentionIO,IO.inputs,IO.address,IO.keyValues,encodedViews,Target.views,Target.get,
      readState,float_map,hr,hd] using ht
  · intro r hr d hd
    have ht:=congrArg (toFloat p.storage.float)
      (binding (2 : Fin 4) (b,h,⟨r,hr⟩,⟨d,hd⟩,()) trivial)
    change readState p s p.V (offset p.v b.val h.val r d)=
      toFloat p.storage.float (mapValue encode p.storage.tile (vals (2 : Fin 4) (b,h,⟨r,hr⟩,⟨d,hd⟩,()))) at ht
    rw [float_map] at ht
    simpa [IO.attentionIO,IO.inputs,IO.address,IO.valueValues,encodedViews,Target.views,Target.get,
      readState,float_map,hr,hd] using ht
  · intro r hr c hc
    have ht:=congrArg (toFloat p.storage.float)
      (binding (3 : Fin 4) (b,h,⟨r,hr⟩,⟨c,hc⟩,()) trivial)
    change readState p s p.Bias (b.val*p.bias.batch+h.val*p.bias.head+r*p.bias.row+c)=
      toFloat p.storage.float (mapValue encode p.storage.tile (vals (3 : Fin 4) (b,h,⟨r,hr⟩,⟨c,hc⟩,()))) at ht
    rw [float_map] at ht
    simpa [IO.attentionIO,IO.inputs,IO.biasAddress,IO.biasValues,encodedViews,Target.views,Target.get,
      readState,float_map,hr,hc] using ht

theorem exterior (M : Algebra β) (dot : Dot β) (p : Parameters) (work : List Program)
    (encode : α → β) (observe : TileIndex (IO.TensorShape p) → β → α)
    (vals : ChannelValues α (IO.inputs p)) (r : RegionName) (off : Nat)
    (outside : (IO.attentionIO M dot p work encode observe).Outside vals r off) :
    Physical.Exterior p r off := by
  by_cases same : r=p.Out
  · right
    intro b hb h hh n hn d hd
    have ht:=outside.1 (0 : Fin 1) (⟨b,hb⟩,⟨h,hh⟩,⟨n,hn⟩,⟨d,hd⟩,()) trivial
    simpa [IO.attentionIO,IO.outputs,IO.address,logicalAddress,same] using ht
  · exact Or.inl same

theorem map_output (storage : TypedSource.Storage) (observe : β → α) (x : β) :
    mapCell observe (.mk storage.tile (ofFloat storage.float x))=
      .mk storage.tile (ofFloat storage.float (observe x)) := by
  cases storage <;> rfl

end
end VeriTile.Bench.Repairs.AttentionKernel.IOMemoryBinding
