import bench.repairs.tritonbench_g.fast_rope_embedding.Proofs.Allocation

namespace VeriTile.Bench.Repairs.FastRopeEmbedding
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs
noncomputable section
set_option maxHeartbeats 4000000

def firstResult {α : Type} [Inhabited α] (api : RopeEmbedding.Memory.API β Store Byte) (embed : β→α)
    (project : α→β) (op : RopeEmbedding.Operations α) (rq rk : RopeEmbedding.Request) (before : Store)
    (p : PairPreparation api embed rq rk before) :=
  RopeEmbedding.Native.entry api embed project op rq Bool.false p.first.state p.first.call

def secondStage {α : Type} [Inhabited α] (api : RopeEmbedding.Memory.API β Store Byte) (embed : β→α)
    (project : α→β) (op : RopeEmbedding.Operations α) (rq rk : RopeEmbedding.Request) (before : Store)
    (p : PairPreparation api embed rq rk before) :=
  p.second (firstResult api embed project op rq rk before p).memory

def secondResult {α : Type} [Inhabited α] (api : RopeEmbedding.Memory.API β Store Byte) (embed : β→α)
    (project : α→β) (op : RopeEmbedding.Operations α) (rq rk : RopeEmbedding.Request) (before : Store)
    (p : PairPreparation api embed rq rk before) :=
  let k := secondStage api embed project op rq rk before p
  RopeEmbedding.Native.entry api embed project op rk Bool.false k.state k.call

structure Result (Store : Type) where
  memory : Store
  q : RopeEmbedding.Native.Tensor
  k : RopeEmbedding.Native.Tensor
  qCache : Cache
  kCache : Cache

def pairResult {α : Type} [Inhabited α] (api : RopeEmbedding.Memory.API β Store Byte) (embed : β→α)
    (project : α→β) (op : RopeEmbedding.Operations α) (rq rk : RopeEmbedding.Request) (before : Store)
    (p : PairPreparation api embed rq rk before) : Result Store :=
  let k := secondStage api embed project op rq rk before p
  ⟨(secondResult api embed project op rq rk before p).memory,p.first.call.output,k.call.output,
    cache rq p.first.call.cosine p.first.call.sine,cache rk k.call.cosine k.call.sine⟩

def check {α : Type} [Inhabited α] (api : RopeEmbedding.Memory.API β Store Byte) (embed : β→α)
    (project : α→β) (op : RopeEmbedding.Operations α) (rq rk : RopeEmbedding.Request) (before : Store)
    (p : PairPreparation api embed rq rk before) :=
  let k := secondStage api embed project op rq rk before p
  inputCheck api embed rq rk before p &&
    RopeEmbedding.Checked.check api embed rq p.first.state p.first.call Bool.false Bool.true &&
    RopeEmbedding.Checked.check api embed rk k.state k.call Bool.false Bool.true

def invoke {α : Type} [Inhabited α] (api : RopeEmbedding.Memory.API β Store Byte) (embed : β→α)
    (project : α→β) (op : RopeEmbedding.Operations α) (rq rk : RopeEmbedding.Request) (before : Store)
    (p : PairPreparation api embed rq rk before) :=
  if check api embed project op rq rk before p then some (pairResult api embed project op rq rk before p) else none

/-- A local read outside the source's write interval is preserved, from the
proved active event bounds. This is independent of any output formula. -/
theorem kernel_keep {α : Type} [Inhabited α] (api : RopeEmbedding.Memory.API β Store Byte) (project : α→β)
    (op : RopeEmbedding.Operations α) (r : RopeEmbedding.Request) (back : Bool) (s : State α) (base : Nat) (memory : Store)
    (address : Nat) (outside : address+4≤base ∨ base+4*r.size≤address) :
    api.read (RopeEmbedding.Memory.run api project op r back s base memory) address=api.read memory address := by
  apply RopeEmbedding.Memory.writes_read_frame
  intro e he
  have bounds := RopeEmbedding.Memory.active_window project op r back s base (RopeEmbedding.schedule r) e he
  omega

theorem first_keeps_k {α : Type} [Inhabited α] (api : RopeEmbedding.Memory.API β Store Byte) (embed : β→α)
    (project : α→β) (op : RopeEmbedding.Operations α) (rq rk : RopeEmbedding.Request) (before : Store)
    (p : PairPreparation api embed rq rk before) (laws : StorageLaws api embed rq rk before p)
    (accepted : inputCheck api embed rq rk before p=Bool.true)
    (b h t c : Nat) (hb : b<rk.batches) (hh : h<rk.heads) (ht : t<rk.sequence) (hc : c<rk.dimension) :
    RopeEmbedding.Native.logicalRead api (firstResult api embed project op rq rk before p).memory p.inputK b h t c=
      RopeEmbedding.Native.logicalRead api before p.inputK b h t c := by
  unfold RopeEmbedding.Native.logicalRead firstResult RopeEmbedding.Native.entry
  rw [kernel_keep api project op rq Bool.false p.first.state p.first.call.output.base p.first.call.memory]
  · exact laws.firstKeepsK b h t c hb hh ht hc
  · rw [laws.firstKAddress b h t c hb hh ht hc]
    have size := laws.firstKBounds b h t c hb hh ht hc
    have sep := laws.firstApartK accepted
    unfold apart at sep
    omega

theorem first_keeps_coefficients {α : Type} [Inhabited α] (api : RopeEmbedding.Memory.API β Store Byte)
    (embed : β→α) (project : α→β) (op : RopeEmbedding.Operations α) (rq rk : RopeEmbedding.Request) (before : Store)
    (p : PairPreparation api embed rq rk before) (t j : Nat) (ht : t<rk.sequence) (hj : j<rk.half) :
    api.read (firstResult api embed project op rq rk before p).memory
      (RopeEmbedding.Checked.tableAddress rk p.first.call.cosine t j)=api.read before (RopeEmbedding.Checked.tableAddress rk p.first.call.cosine t j) ∧
    api.read (firstResult api embed project op rq rk before p).memory
      (RopeEmbedding.Checked.tableAddress rk p.first.call.sine t j)=api.read before (RopeEmbedding.Checked.tableAddress rk p.first.call.sine t j) := by
  have half : rq.half=rk.half := by simp only [RopeEmbedding.Request.half,p.geometry.2.2]
  have bound : t*rk.half+j<rq.sequence*rq.half := by
    rw [p.geometry.2.1,half]
    have h := Nat.mul_le_mul_right rk.half (show t+1≤rk.sequence by omega)
    rw [Nat.add_mul,Nat.one_mul] at h
    omega
  have cosine := p.first.call.separateCosine
  have sine := p.first.call.separateSine
  constructor
  · change api.read (RopeEmbedding.Memory.run api project op rq Bool.false p.first.state p.first.call.output.base p.first.call.memory)
      (p.first.call.cosine.base+4*(t*rk.half+j))=_
    rw [kernel_keep api project op rq Bool.false p.first.state p.first.call.output.base p.first.call.memory _ (by omega),
      p.first.call.preservedCosine _ bound,p.first.entered]
    rfl
  · change api.read (RopeEmbedding.Memory.run api project op rq Bool.false p.first.state p.first.call.output.base p.first.call.memory)
      (p.first.call.sine.base+4*(t*rk.half+j))=_
    rw [kernel_keep api project op rq Bool.false p.first.state p.first.call.output.base p.first.call.memory _ (by omega),
      p.first.call.preservedSine _ bound,p.first.entered]
    rfl

def Caller {α : Type} [Inhabited α] (api : RopeEmbedding.Memory.API β Store Byte) (embed : β→α)
    (rq rk : RopeEmbedding.Request) (before : Store) (p : PairPreparation api embed rq rk before) (address : Nat) : Prop :=
  (p.first.call.input.allocationBase≤address ∧ address<p.first.call.input.allocationEnd) ∨
  (p.inputK.allocationBase≤address ∧ address<p.inputK.allocationEnd) ∨
  (p.first.call.cosineInput.allocationBase≤address ∧ address<p.first.call.cosineInput.allocationEnd) ∨
  (p.first.call.sineInput.allocationBase≤address ∧ address<p.first.call.sineInput.allocationEnd)

theorem second_keeps_first {α : Type} [Inhabited α] (api : RopeEmbedding.Memory.API β Store Byte) (embed : β→α)
    (project : α→β) (op : RopeEmbedding.Operations α) (rq rk : RopeEmbedding.Request) (before : Store)
    (p : PairPreparation api embed rq rk before) (laws : StorageLaws api embed rq rk before p)
    (accepted : inputCheck api embed rq rk before p=Bool.true)
    (b h t c : Nat) (hb : b<rq.batches) (hh : h<rq.heads) (ht : t<rq.sequence) (hc : c<rq.dimension) :
    RopeEmbedding.Native.logicalRead api (secondResult api embed project op rq rk before p).memory p.first.call.output b h t c=
      RopeEmbedding.Native.logicalRead api (firstResult api embed project op rq rk before p).memory p.first.call.output b h t c := by
  let q := firstResult api embed project op rq rk before p
  let k := secondStage api embed project op rq rk before p
  change api.read (RopeEmbedding.Memory.run api project op rk Bool.false k.state k.call.output.base k.call.memory)
    (RopeEmbedding.Native.logicalAddress p.first.call.output b h t c)=_
  rw [kernel_keep api project op rk Bool.false k.state k.call.output.base k.call.memory]
  · exact laws.secondKeepsQ q.memory b h t c hb hh ht hc
  · rw [p.first.call.outputAddress b h t c hb hh ht hc]
    have bound := RopeEmbedding.offset_lt rq (b*rq.sequence+t) h c (RopeEmbedding.Native.row_bound rq b t hb ht) hh hc
    have sep := laws.outputApart accepted q.memory
    change apart p.first.call.output.base rq.size k.call.output.base rk.size at sep
    unfold apart at sep
    omega

theorem pair_frame {α : Type} [Inhabited α] (api : RopeEmbedding.Memory.API β Store Byte) (embed : β→α)
    (project : α→β) (op : RopeEmbedding.Operations α) (rq rk : RopeEmbedding.Request) (before : Store)
    (p : PairPreparation api embed rq rk before) (laws : StorageLaws api embed rq rk before p)
    (address : Nat) (caller : Caller api embed rq rk before p address)
    (outsideQ : address<p.first.call.output.base ∨ p.first.call.output.base+4*rq.size≤address)
    (outsideK : address<(secondStage api embed project op rq rk before p).call.output.base ∨
      (secondStage api embed project op rq rk before p).call.output.base+4*rk.size≤address) :
    api.byte (pairResult api embed project op rq rk before p).memory address=api.byte before address := by
  let q := firstResult api embed project op rq rk before p
  let k := secondStage api embed project op rq rk before p
  change api.byte (RopeEmbedding.Memory.run api project op rk Bool.false k.state k.call.output.base k.call.memory) address=_
  have prepared := laws.secondFrame q.memory address caller
  change api.byte k.call.memory address=api.byte q.memory address at prepared
  rw [RopeEmbedding.Memory.byte_frame api project op rk Bool.false k.state k.call.output.base k.call.memory address outsideK,prepared]
  change api.byte (RopeEmbedding.Memory.run api project op rq Bool.false p.first.state p.first.call.output.base p.first.call.memory) address=_
  rw [RopeEmbedding.Memory.byte_frame api project op rq Bool.false p.first.state p.first.call.output.base p.first.call.memory address outsideQ,
    laws.firstFrame address caller]

#axiomsClean first_keeps_k
#axiomsClean first_keeps_coefficients
#axiomsClean second_keeps_first
#axiomsClean pair_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastRopeEmbedding
