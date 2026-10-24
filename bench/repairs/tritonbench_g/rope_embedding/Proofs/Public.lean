import bench.repairs.tritonbench_g.rope_embedding.Proofs.Binding
import bench.repairs.tritonbench_g.rope_embedding.Proofs.Guards
namespace VeriTile.Bench.Repairs.RopeEmbedding.Checked
open VeriTile.Triton FP.Structural
open Numerics (Word Tagged leaf)
noncomputable section
set_option maxHeartbeats 3000000

/-- Physical returned object, original logical indexing, complete four-byte
readback and exact caller frame outside the output interval. -/
structure Output {α : Type} [Inhabited α] (api : Memory.API β Store Byte) (embed : β→α)
    (r : Request) (s : State α) (call : Native.Preparation api embed r s)
    (value : Nat→Nat→Nat→Nat→β) (out : Native.Returned Store) : Prop where
  returned : out.tensor=call.output
  shape : out.tensor.shape=[r.batches,r.heads,r.sequence,r.dimension]
  dtype : out.tensor.dtype="float32" ∧ out.tensor.elementBytes=4
  device : out.tensor.device=call.input.device
  logical : ∀ b h t c,b<r.batches → h<r.heads → t<r.sequence → c<r.dimension →
    Native.logicalRead api out.memory out.tensor b h t c=value b h t c
  bytes : ∀ b h t c,b<r.batches → h<r.heads → t<r.sequence → c<r.dimension → ∀ k : Fin 4,
    api.byte out.memory (Native.logicalAddress out.tensor b h t c+k.val)=api.encode (value b h t c) k
  frame : ∀ b,call.caller b → (b<call.output.base ∨ call.output.base+4*r.size≤b) →
    api.byte out.memory b=api.byte call.before b

def RealOutput (api : Memory.API ℝ Store Byte) (r : Request) (back : Bool) (s : State ℝ)
    (call : Native.Preparation api id r s) (out : Native.Returned Store) : Prop :=
  Output api id r s call (mathematicalTarget api r back call.before call.input call.cosine call.sine) out ∧
    Native.SourcePrograms api id id ideal r back s call ∧ Native.IndexABI api id r s call

def FP32Output (api : Memory.API Word Store (BitVec 8)) (p : Numerics.Primitives)
    (r : Request) (back : Bool) (s : State Tagged) (call : Native.Preparation api leaf r s)
    (out : Native.Returned Store) : Prop :=
  Output api leaf r s call (machineTarget api p r back call.before call.input call.cosine call.sine) out ∧
    Native.SourcePrograms api leaf Tagged.word (Numerics.operations p) r back s call ∧
    Native.IndexABI api leaf r s call ∧ (∀ w k,api.encode w k=Numerics.wordByte w k)

theorem whole_real (api : Memory.API ℝ Store Byte) (r : Request) (back : Bool) (s : State ℝ)
    (call : Native.Preparation api id r s) (clone squeeze : Bool)
    (accepted : check api id r s call clone squeeze=Bool.true) :
    RealOutput api r back s call (Native.entry api id id ideal r back s call) := by
  have old := Native.output_contract api id id ideal r back s call (fun _ => rfl)
  have values : ∀ b h t c,b<r.batches → h<r.heads → t<r.sequence → c<r.dimension →
      Native.logicalRead api (Native.entry api id id ideal r back s call).memory call.output b h t c=
        mathematicalTarget api r back call.before call.input call.cosine call.sine b h t c := by
    intro b h t c hb hh ht hc
    have source := old.logical b h t c hb hh ht hc
    rw [old.returned] at source
    rw [source]
    change scalarResult ideal r back _ _ _ _ h c=_
    rw [scalar_binding api id ideal r back s call b h t c hb hh ht hc]
    exact ideal_logical api r back call.before call.input call.cosine call.sine b h t c
  refine ⟨⟨old.returned,old.shape,old.dtype,old.device,values,?_,old.frame⟩,
    Native.source_programs api id id ideal r back s call,indexABI api id r s call clone squeeze accepted⟩
  intro b h t c hb hh ht hc k
  rw [api.read_bytes]
  exact congrArg (fun value => api.encode value k) (values b h t c hb hh ht hc)

theorem whole_fp32 (api : Memory.API Word Store (BitVec 8)) (p : Numerics.Primitives)
    (r : Request) (back : Bool) (s : State Tagged) (call : Native.Preparation api leaf r s)
    (clone squeeze : Bool) (codec : ∀ w k,api.encode w k=Numerics.wordByte w k)
    (accepted : check api leaf r s call clone squeeze=Bool.true) :
    FP32Output api p r back s call (Native.entry api leaf Tagged.word (Numerics.operations p) r back s call) := by
  have old := Native.output_contract api leaf Tagged.word (Numerics.operations p) r back s call (fun _ => rfl)
  have values : ∀ b h t c,b<r.batches → h<r.heads → t<r.sequence → c<r.dimension →
      Native.logicalRead api (Native.entry api leaf Tagged.word (Numerics.operations p) r back s call).memory
        call.output b h t c=machineTarget api p r back call.before call.input call.cosine call.sine b h t c := by
    intro b h t c hb hh ht hc
    have source := old.logical b h t c hb hh ht hc
    rw [old.returned] at source
    rw [source]
    rw [scalar_binding api leaf (Numerics.operations p) r back s call b h t c hb hh ht hc]
    exact native_logical api p r back call.before call.input call.cosine call.sine b h t c
  refine ⟨⟨old.returned,old.shape,old.dtype,old.device,values,?_,old.frame⟩,
    Native.source_programs api leaf Tagged.word (Numerics.operations p) r back s call,
    indexABI api leaf r s call clone squeeze accepted,codec⟩
  intro b h t c hb hh ht hc k
  rw [api.read_bytes]
  exact congrArg (fun value => api.encode value k) (values b h t c hb hh ht hc)

def forward {α : Type} [Inhabited α] (api : Memory.API β Store Byte) (embed : β→α) (project : α→β)
    (op : Operations α) (r : Request) (s : State α) (call : Native.Preparation api embed r s) :=
  if check api embed r s call Bool.true Bool.true then some (Native.forward api embed project op r s call) else none

def backward {α : Type} [Inhabited α] (api : Memory.API β Store Byte) (embed : β→α) (project : α→β)
    (op : Operations α) (r : Request) (s : State α) (call : Native.Preparation api embed r s)
    (groups block warps : Nat) :=
  if check api embed r s call Bool.false Bool.false && cacheCheck r groups block warps then
    Native.backward api embed project op r s call groups block warps else none

/-- Six returned fields and the full unchanged caller backing frame of the
original forward-only clone, in addition to the complete output property. -/
def ForwardResult {α : Type} [Inhabited α] (api : Memory.API β Store Byte) (embed : β→α)
    (r : Request) (s : State α) (call : Native.Preparation api embed r s)
    (property : Native.Returned Store → Prop) (out : Native.ForwardReturned Store) : Prop :=
  property out.toReturned ∧ out.cosine=call.cosine ∧ out.sine=call.sine ∧
    (out.groups,out.block,out.warps)=(r.groups,r.block,r.warps) ∧
    ∀ b,call.caller b → api.byte out.memory b=api.byte call.before b

theorem forward_real (api : Memory.API ℝ Store Byte) (r : Request) (s : State ℝ)
    (call : Native.Preparation api id r s) (accepted : check api id r s call Bool.true Bool.true=Bool.true) :
    ∃ out,forward api id id ideal r s call=some out ∧
      ForwardResult api id r s call (RealOutput api r Bool.false s call) out := by
  have whole := whole_real api r Bool.false s call Bool.true Bool.true accepted
  refine ⟨Native.forward api id id ideal r s call,by simp [forward,accepted],whole,rfl,rfl,rfl,?_⟩
  intro b hb
  exact whole.1.frame b hb (forward_fresh api id r s call accepted b hb)

theorem backward_real (api : Memory.API ℝ Store Byte) (r : Request) (s : State ℝ)
    (call : Native.Preparation api id r s) (groups block warps : Nat)
    (accepted : check api id r s call Bool.false Bool.false=Bool.true)
    (cached : cacheCheck r groups block warps=Bool.true) :
    ∃ out,backward api id id ideal r s call groups block warps=some out ∧ RealOutput api r Bool.true s call out := by
  exact ⟨Native.entry api id id ideal r Bool.true s call,
    by simp [backward,accepted,cached,Native.backward,cache_admitted r groups block warps cached],
    whole_real api r Bool.true s call Bool.false Bool.false accepted⟩

theorem forward_fp32 (api : Memory.API Word Store (BitVec 8)) (p : Numerics.Primitives)
    (r : Request) (s : State Tagged) (call : Native.Preparation api leaf r s)
    (codec : ∀ w k,api.encode w k=Numerics.wordByte w k)
    (accepted : check api leaf r s call Bool.true Bool.true=Bool.true) :
    ∃ out,forward api leaf Tagged.word (Numerics.operations p) r s call=some out ∧
      ForwardResult api leaf r s call (FP32Output api p r Bool.false s call) out := by
  have whole := whole_fp32 api p r Bool.false s call Bool.true Bool.true codec accepted
  refine ⟨Native.forward api leaf Tagged.word (Numerics.operations p) r s call,
    by simp [forward,accepted],whole,rfl,rfl,rfl,?_⟩
  intro b hb
  exact whole.1.frame b hb (forward_fresh api leaf r s call accepted b hb)

theorem backward_fp32 (api : Memory.API Word Store (BitVec 8)) (p : Numerics.Primitives)
    (r : Request) (s : State Tagged) (call : Native.Preparation api leaf r s) (groups block warps : Nat)
    (codec : ∀ w k,api.encode w k=Numerics.wordByte w k)
    (accepted : check api leaf r s call Bool.false Bool.false=Bool.true)
    (cached : cacheCheck r groups block warps=Bool.true) :
    ∃ out,backward api leaf Tagged.word (Numerics.operations p) r s call groups block warps=some out ∧
      FP32Output api p r Bool.true s call out := by
  exact ⟨Native.entry api leaf Tagged.word (Numerics.operations p) r Bool.true s call,
    by simp [backward,accepted,cached,Native.backward,cache_admitted r groups block warps cached],
    whole_fp32 api p r Bool.true s call Bool.false Bool.false codec accepted⟩

#axiomsClean whole_real
#axiomsClean whole_fp32
#axiomsClean forward_real
#axiomsClean backward_real
#axiomsClean forward_fp32
#axiomsClean backward_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RopeEmbedding.Checked
