import bench.repairs.tritonbench_g.rope_embedding.Proofs.Composition
namespace VeriTile.Bench.Repairs.RopeEmbedding.Native
open VeriTile.Triton FP.Structural
noncomputable section

def launchSchedule (r : Request) : List (Location r) := if r.size=0 ∨ r.half=0 then [] else schedule r

def SourcePrograms {α : Type} [Inhabited α] (api : Memory.API β Store Byte) (embed : β→α) (project : α→β)
    (op : Operations α) (r : Request) (back : Bool) (s : State α) (call : Preparation api embed r s) : Prop :=
  ∀ l∈launchSchedule r,∃ final,FP.Structural.exec op.backend (Source.source (parameters r) back)
    (programState (snapshot api embed r s call (prefixMemory api embed project op r back s call (row l) (group l)))
      (row l) (group l))=some final ∧
    (active l → final.mem "Q" (r.offset (row l) (head l) (column l))=.mk .real (sourceValue op r back s l))

structure OutputContract {α : Type} [Inhabited α] (api : Memory.API β Store Byte) (embed : β→α)
    (r : Request) (s : State α) (call : Preparation api embed r s) (value : Nat→Nat→Nat→β)
    (out : Returned Store) : Prop where
  returned : out.tensor=call.output
  shape : out.tensor.shape=[r.batches,r.heads,r.sequence,r.dimension]
  dtype : out.tensor.dtype="float32" ∧ out.tensor.elementBytes=4
  device : out.tensor.device=call.input.device
  logical : ∀ b h m c,b<r.batches → h<r.heads → m<r.sequence → c<r.dimension →
    logicalRead api out.memory out.tensor b h m c=value (b*r.sequence+m) h c
  bytes : ∀ b h m c,b<r.batches → h<r.heads → m<r.sequence → c<r.dimension → ∀ k : Fin 4,
    api.byte out.memory (logicalAddress out.tensor b h m c+k.val)=api.encode (value (b*r.sequence+m) h c) k
  frame : ∀ b,call.caller b → (b<call.output.base ∨ call.output.base+4*r.size≤b) →
    api.byte out.memory b=api.byte call.before b

structure ForwardContract {α : Type} [Inhabited α] (api : Memory.API β Store Byte) (embed : β→α)
    (r : Request) (s : State α) (call : Preparation api embed r s) (value : Nat→Nat→Nat→β)
    (out : ForwardReturned Store) : Prop extends OutputContract api embed r s call value out.toReturned where
  cosineIdentity : out.cosine=call.cosine
  sineIdentity : out.sine=call.sine
  cache : (out.groups,out.block,out.warps)=(r.groups,r.block,r.warps)

theorem source_programs {α : Type} [Inhabited α] (api : Memory.API β Store Byte) (embed : β→α) (project : α→β)
    (op : Operations α) (r : Request) (back : Bool) (s : State α) (call : Preparation api embed r s) :
    SourcePrograms api embed project op r back s call := by
  intro l hl
  have hs : l∈schedule r := by
    unfold launchSchedule at hl
    split at hl
    · simp at hl
    · exact hl
  exact source_execution_prefix api embed project op r back s call l hs

theorem output_contract {α : Type} [Inhabited α] (api : Memory.API β Store Byte) (embed : β→α) (project : α→β)
    (op : Operations α) (r : Request) (back : Bool) (s : State α) (call : Preparation api embed r s)
    (roundtrip : ∀ w,project (embed w)=w) :
    OutputContract api embed r s call (fun a h c => project (scalarResult op r back
      (fun o => (s.mem "Q" o).read .real) (fun o => (s.mem "C" o).read .real) (fun o => (s.mem "S" o).read .real) a h c))
      (entry api embed project op r back s call) := by
  obtain ⟨hr,hs,hv,hb,hf⟩ := whole api embed project op r back s call roundtrip
  exact ⟨hr,hs,call.outputType,call.devices.1,hv,hb,hf⟩

#axiomsClean source_programs
#axiomsClean output_contract
end
end VeriTile.Bench.Repairs.RopeEmbedding.Native
