import bench.repairs.tritonbench_g.rope_embedding.Proofs.Public

namespace VeriTile.Bench.Repairs.FastRopeEmbedding
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs

/-- Metadata for the actual inner [B,S,H,D] view returned by the autograd
callback and its outer transpose. No data is copied by this binding. -/
structure InnerView (r : RopeEmbedding.Request) (outer : RopeEmbedding.Native.Tensor) where
  tensor : RopeEmbedding.Native.Tensor
  shape : tensor.shape=[r.batches,r.sequence,r.heads,r.dimension]
  storage : tensor.storage=outer.storage ∧ tensor.base=outer.base ∧ tensor.device=outer.device
  address : ∀ b h t c,b<r.batches → h<r.heads → t<r.sequence → c<r.dimension →
    RopeEmbedding.Native.logicalAddress tensor b t h c=RopeEmbedding.Native.logicalAddress outer b h t c

structure Cache where
  groups : Nat
  block : Nat
  warps : Nat
  cosine : RopeEmbedding.Native.Tensor
  sine : RopeEmbedding.Native.Tensor

def cache (r : RopeEmbedding.Request) (cosine sine : RopeEmbedding.Native.Tensor) : Cache :=
  ⟨r.groups,r.block,r.warps,cosine,sine⟩

/-- An ordinary reshape/squeeze preparation, indexed by the memory at which it
is called. This makes the second call's entry state the first call's result by
construction rather than by an assumed whole-output equation. -/
structure Stage {α : Type} [Inhabited α] (api : RopeEmbedding.Memory.API β Store Byte) (embed : β→α)
    (r : RopeEmbedding.Request) (before : Store) where
  state : State α
  call : RopeEmbedding.Native.Preparation api embed r state
  entered : call.before=before
  inner : InnerView r call.output

structure PairPreparation {α : Type} [Inhabited α] (api : RopeEmbedding.Memory.API β Store Byte)
    (embed : β→α) (rq rk : RopeEmbedding.Request) (before : Store) where
  first : Stage api embed rq before
  inputK : RopeEmbedding.Native.Tensor
  second : (after : Store) → Stage api embed rk after
  secondInput : ∀ after,(second after).call.input=inputK
  secondCoefficients : ∀ after,
    (second after).call.cosine.base=first.call.cosine.base ∧
    (second after).call.sine.base=first.call.sine.base
  geometry : rq.batches=rk.batches ∧ rq.sequence=rk.sequence ∧ rq.dimension=rk.dimension

def inputCheck {α : Type} [Inhabited α] (api : RopeEmbedding.Memory.API β Store Byte)
    (embed : β→α) (rq rk : RopeEmbedding.Request) (before : Store) (p : PairPreparation api embed rq rk before) :=
  RopeEmbedding.Checked.separate p.first.call.input p.inputK &&
    decide (p.first.call.input.device=p.inputK.device) &&
    decide (rq.batches=rk.batches ∧ rq.sequence=rk.sequence ∧ rq.dimension=rk.dimension)

def apart (base size other count : Nat) : Prop := base+4*size≤other ∨ other+4*count≤base

/-- Ordinary allocation/reshape laws. Input allocation checks plus reuse or
fresh allocation give physical separation. Preparation preserves reads and
caller bytes outside newly allocated storage; no kernel result is stipulated. -/
structure StorageLaws {α : Type} [Inhabited α] (api : RopeEmbedding.Memory.API β Store Byte)
    (embed : β→α) (rq rk : RopeEmbedding.Request) (before : Store) (p : PairPreparation api embed rq rk before) : Prop where
  firstInputLower : p.first.call.input.allocationBase≤p.first.call.input.base ∨ rq.size=0
  secondInputLower : p.inputK.allocationBase≤p.inputK.base ∨ rk.size=0
  firstOutputLower : p.first.call.output.allocationBase≤p.first.call.output.base ∨ rq.size=0
  secondOutputLower : ∀ after,(p.second after).call.output.allocationBase≤(p.second after).call.output.base ∨ rk.size=0
  firstEmpty : rq.size=0 → p.first.call.output.base=0
  secondInputEmpty : rk.size=0 → p.inputK.base=0
  secondEmpty : ∀ after,rk.size=0 → (p.second after).call.output.base=0
  firstFreshK : p.first.call.reused=Bool.false →
    p.first.call.output.allocationEnd≤p.inputK.allocationBase ∨
    p.inputK.allocationEnd≤p.first.call.output.allocationBase
  secondFreshQ : ∀ after,(p.second after).call.reused=Bool.false →
    (p.second after).call.output.allocationEnd≤p.first.call.output.allocationBase ∨
    p.first.call.output.allocationEnd≤(p.second after).call.output.allocationBase
  firstKeepsK : ∀ b h t c,b<rk.batches → h<rk.heads → t<rk.sequence → c<rk.dimension →
    api.read p.first.call.memory (RopeEmbedding.Native.logicalAddress p.inputK b h t c)=
      api.read before (RopeEmbedding.Native.logicalAddress p.inputK b h t c)
  secondKeepsQ : ∀ after b h t c,b<rq.batches → h<rq.heads → t<rq.sequence → c<rq.dimension →
    api.read (p.second after).call.memory (RopeEmbedding.Native.logicalAddress p.first.call.output b h t c)=
      api.read after (RopeEmbedding.Native.logicalAddress p.first.call.output b h t c)
  firstKAddress : ∀ b h t c,b<rk.batches → h<rk.heads → t<rk.sequence → c<rk.dimension →
    RopeEmbedding.Native.logicalAddress p.inputK b h t c=p.inputK.base+4*rk.external b h t c
  firstKBounds : ∀ b h t c,b<rk.batches → h<rk.heads → t<rk.sequence → c<rk.dimension →
    rk.external b h t c<rk.size
  firstFrame : ∀ address,
    ((p.first.call.input.allocationBase≤address ∧ address<p.first.call.input.allocationEnd) ∨
      (p.inputK.allocationBase≤address ∧ address<p.inputK.allocationEnd) ∨
      (p.first.call.cosineInput.allocationBase≤address ∧ address<p.first.call.cosineInput.allocationEnd) ∨
      (p.first.call.sineInput.allocationBase≤address ∧ address<p.first.call.sineInput.allocationEnd)) →
    api.byte p.first.call.memory address=api.byte before address
  secondFrame : ∀ after address,
    ((p.first.call.input.allocationBase≤address ∧ address<p.first.call.input.allocationEnd) ∨
      (p.inputK.allocationBase≤address ∧ address<p.inputK.allocationEnd) ∨
      (p.first.call.cosineInput.allocationBase≤address ∧ address<p.first.call.cosineInput.allocationEnd) ∨
      (p.first.call.sineInput.allocationBase≤address ∧ address<p.first.call.sineInput.allocationEnd)) →
    api.byte (p.second after).call.memory address=api.byte after address

end VeriTile.Bench.Repairs.FastRopeEmbedding
