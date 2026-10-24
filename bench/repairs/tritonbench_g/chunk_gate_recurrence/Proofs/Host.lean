import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Preparation

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.Host
open Checked Binding Machine
open VeriTile.Bench.Repairs.SwigluTriton.Memory (Key)
noncomputable section

structure Forward (M : Type) where
  output : Tensor
  savedOutput : Tensor
  savedGate : Tensor
  launched : Bool
  memory : M

structure Backward (M : Type) where
  dx : Tensor
  dgate : Tensor
  dseed : Option Tensor
  launched : Bool
  memory : M

/-- Closed full forward source followed by its proven physical write normal
form. This is an interpreter definition, never an external result callback. -/
def forward {M α β : Type} (api : API M α β) (ops : Ops α) (c : Shape) (seed : Bool)
    (runtime : Runtime c seed) (m : M) : Forward M :=
  ⟨runtime.tensor .output,runtime.tensor .output,runtime.tensor .gate,decide (0<c.N),
   stageCommit api c seed runtime Bool.true
     (Source.forwardGrid ops c seed (mirror api c seed runtime m) c.programs) m⟩

def backward {M α β : Type} (api : API M α β) (ops : Ops α) (c : Shape) (seed : Bool)
    (runtime : Runtime c seed) (trees : Trees c) (m : M) : Backward M :=
  ⟨runtime.tensor .dx,runtime.tensor .dgate,if seed then some (runtime.tensor .dseed) else none,
   decide (0<c.N),stageCommit api c seed runtime Bool.false
     (Source.backwardRun ops c trees seed (mirror api c seed runtime m)) m⟩

def nativeBackward {M β : Type} (api : API M Word β) (isa : ISA) (c : Shape) (seed : Bool)
    (runtime : Runtime c seed) (trees : Trees c) (m : M) : Backward M :=
  ⟨runtime.tensor .dx,runtime.tensor .dgate,if seed then some (runtime.tensor .dseed) else none,
   decide (0<c.N),stageCommit api c seed runtime Bool.false
     (SplitNative.backward isa c trees seed (mirror api c seed runtime m)) m⟩

/-- Ordinary no-write semantics across backward's incoming contiguous copy,
fresh allocations, and saved-tensor access. It states a local storage law for
the two saved allocations, with no numerical formula or execution receipt. -/
structure SavedStorage {M α β : Type} (api : API M α β) (runtime : Runtime c seed)
    (before after : M) : Prop where
  read : ∀ r,r=.output ∨ r=.gate → ∀ key : Key,
    key.owner=(runtime.tensor r).device →
    (runtime.tensor r).allocationBase ≤ key.address →
    key.address+4 ≤ (runtime.tensor r).allocationBase+(runtime.tensor r).allocationBytes →
      api.read after key=api.read before key

theorem forward_read {M α β : Type} (api : API M α β) (ops : Ops α) (c : Shape) (seed : Bool)
    (runtime : Runtime c seed) (accepted : check c seed runtime=Bool.true) (m : M)
    (p t j : Nat) (hp : p<c.programs) (ht : t<c.N) (hj : j<1024) :
    api.read (forward api ops c seed runtime m).memory (key c seed runtime ⟨.output,p,t,j⟩)=
      Reference.forward ops c (inputsOf (mirror api c seed runtime m)) seed p t j := by
  change api.read (stageCommit _ _ _ _ _ _ _) _=_
  rw [stage_read api c seed runtime accepted Bool.true _ m ⟨.output,p,t,j⟩
    ((stage_mem c seed Bool.true ⟨.output,p,t,j⟩).mpr ⟨Or.inl ⟨rfl,hp,ht,hj⟩,by simp⟩)]
  exact Source.forwardGrid_output ops c seed _ p t j hp ht hj

theorem saved_read {M α β : Type} (api : API M α β) (ops : Ops α) (c : Shape) (seed : Bool)
    (runtime : Runtime c seed) (accepted : check c seed runtime=Bool.true) (initial prepared : M)
    (saved : SavedStorage api runtime (forward api ops c seed runtime initial).memory prepared)
    (p t j : Nat) (hp : p<c.programs) (ht : t<c.N) (hj : j<1024) :
    mirror api c seed runtime prepared .output p t j=
      Reference.forward ops c (inputsOf (mirror api c seed runtime initial)) seed p t j := by
  have inside := key_contained c seed runtime accepted ⟨.output,p,t,j⟩ ⟨hp,ht,hj⟩
  rw [mirror,saved.read .output (Or.inl rfl) _ rfl inside.1 inside.2]
  exact forward_read api ops c seed runtime accepted initial p t j hp ht hj

theorem backward_read {M α β : Type} (api : API M α β) (ops : Ops α) (c : Shape) (seed : Bool)
    (runtime : Runtime c seed) (accepted : check c seed runtime=Bool.true) (trees : Trees c) (m : M)
    (x : Cell) (write : Source.writes c seed x.region x.program x.time x.lane) (notOutput : x.region≠.output) :
    api.read (backward api ops c seed runtime trees m).memory (key c seed runtime x)=
      Source.backwardTarget ops c trees seed (mirror api c seed runtime m) x.region x.program x.time x.lane := by
  change api.read (stageCommit _ _ _ _ _ _ _) _=_
  rw [stage_read api c seed runtime accepted Bool.false _ m x
    ((stage_mem c seed Bool.false x).mpr ⟨write,notOutput⟩),Source.backward_complete]

theorem native_read {M β : Type} (api : API M Word β) (isa : ISA)
    (laws : LocalConformance isa allOperands) (c : Shape) (seed : Bool)
    (runtime : Runtime c seed) (accepted : check c seed runtime=Bool.true) (trees : Trees c) (m : M)
    (x : Cell) (write : Source.writes c seed x.region x.program x.time x.lane) (notOutput : x.region≠.output) :
    api.read (nativeBackward api isa c seed runtime trees m).memory (key c seed runtime x)=
      Source.backwardTarget (wordOps SourceFP.scalar) c trees seed (mirror api c seed runtime m)
        x.region x.program x.time x.lane := by
  change api.read (stageCommit _ _ _ _ _ _ _) _=_
  rw [stage_read api c seed runtime accepted Bool.false _ m x
    ((stage_mem c seed Bool.false x).mpr ⟨write,notOutput⟩),SplitNative.backward_complete isa laws]

theorem caller_outside (c : Shape) (seed : Bool) (runtime : Runtime c seed)
    (accepted : check c seed runtime=Bool.true) (a : Allocation) (live : a∈runtime.callerLive)
    (owner address : Nat) (lo : a.storageBase ≤ address) (hi : (address : Int)<a.storageBase+a.byteLength)
    (phase : Bool) : StageOutside c seed runtime phase owner address := by
  intro x hx cover
  have member := stage_subset c seed phase x hx
  have active := active_written c seed _ _ _ _ ((cells_mem c seed x).mp member)
  have inside := key_contained c seed runtime accepted x active.2
  have fresh := runtime.freshCaller x.region active.1 a live
  change ((runtime.tensor x.region).allocationBase : Int)+(runtime.tensor x.region).allocationBytes ≤ a.storageBase ∨
    a.storageBase+a.byteLength ≤ (runtime.tensor x.region).allocationBase at fresh
  rcases cover with ⟨_,coverLo,coverHi⟩
  rcases fresh with before|after <;> omega

theorem forward_gate {M α β : Type} (api : API M α β) (ops : Ops α) (c : Shape) (seed : Bool)
    (runtime : Runtime c seed) (accepted : check c seed runtime=Bool.true) (m : M)
    (h t : Nat) (hh : h<c.heads) (ht : t<c.N) :
    api.read (forward api ops c seed runtime m).memory (key c seed runtime ⟨.gate,h,t,0⟩)=
      api.read m (key c seed runtime ⟨.gate,h,t,0⟩) := by
  apply VeriTile.Bench.Repairs.SwigluTriton.Memory.read_frame
  intro x hx
  have facts := (stage_mem c seed Bool.true x).mp hx
  have region : x.region=.output := by simpa using facts.2
  have active := (active_written c seed _ _ _ _ facts.1).2
  have writer := key_contained c seed runtime accepted x active
  have reader := key_contained c seed runtime accepted ⟨.gate,h,t,0⟩ ⟨hh,ht,rfl⟩
  dsimp only at reader
  have separate := runtime.freshForward .gate (Or.inr (Or.inl rfl))
  change ((runtime.tensor .output).allocationBase : Int)+(runtime.tensor .output).allocationBytes ≤
      (runtime.tensor .gate).allocationBase ∨
    ((runtime.tensor .gate).allocationBase : Int)+(runtime.tensor .gate).allocationBytes ≤
      (runtime.tensor .output).allocationBase at separate
  rw [region] at writer
  rcases separate with before|after
  · exact Or.inr (Or.inr (by omega))
  · exact Or.inr (Or.inl (by omega))

structure Invocation {M α β : Type} (api : API M α β) (ops : Ops α)
    (c : Shape) (seed : Bool) (runtime : Runtime c seed) where
  forwardCaller : Region → Tensor
  forwardBefore : M
  forwardPrepared : M
  forwardCopies : Preparation.Copies api c seed Bool.true runtime forwardCaller forwardBefore forwardPrepared
  forwardLive : ∀ r,Preparation.ReadRole Bool.true r → present c seed r=Bool.true →
    allocation (forwardCaller r)∈runtime.callerLive
  backwardCaller : Region → Tensor
  backwardBefore : M
  backwardPrepared : M
  backwardCopies : Preparation.Copies api c seed Bool.false runtime backwardCaller backwardBefore backwardPrepared
  /-- Each backward result allocation is created after the incoming gradient
  argument exists; the earlier forward output is allowed to be that argument. -/
  backwardFresh : ∀ r,Machine.Written r → r≠.output →
    ∀ q,Preparation.ReadRole Bool.false q → present c seed q=Bool.true →
      (allocation (runtime.tensor r)).Disjoint (allocation (backwardCaller q))
  /-- `ctx.saved_tensors[1]` is the same prepared gate Tensor. -/
  savedGateIdentity : backwardCaller .gate=runtime.tensor .gate
  savedStorage : SavedStorage api runtime (forward api ops c seed runtime forwardPrepared).memory backwardPrepared

def Invocation.forwardInputs {M α β : Type} {api : API M α β} {ops : Ops α}
    {c : Shape} {seed : Bool} {runtime : Runtime c seed} (call : Invocation api ops c seed runtime) : Inputs α :=
  inputsOf (Preparation.callerMemory api c call.forwardCaller call.forwardBefore)

def Invocation.backwardInputs {M α β : Type} {api : API M α β} {ops : Ops α}
    {c : Shape} {seed : Bool} {runtime : Runtime c seed} (call : Invocation api ops c seed runtime) : Inputs α :=
  inputsOf (Preparation.callerMemory api c call.backwardCaller call.backwardBefore)

theorem invocation_saved {M α β : Type} (api : API M α β) (ops : Ops α) (c : Shape) (seed : Bool)
    (runtime : Runtime c seed) (accepted : check c seed runtime=Bool.true) (call : Invocation api ops c seed runtime)
    (p t j : Nat) (hp : p<c.programs) (ht : t<c.N) (hj : j<1024) :
    mirror api c seed runtime call.backwardPrepared .output p t j=
      Reference.forward ops c call.forwardInputs seed p t j := by
  rw [saved_read api ops c seed runtime accepted call.forwardPrepared call.backwardPrepared
    call.savedStorage p t j hp ht hj]
  exact Preparation.forward_reads api ops c seed runtime call.forwardCaller call.forwardBefore
    call.forwardPrepared call.forwardCopies p t j hp ht hj

theorem invocation_gate {M α β : Type} (api : API M α β) (ops : Ops α) (c : Shape) (seed : Bool)
    (runtime : Runtime c seed) (accepted : check c seed runtime=Bool.true) (call : Invocation api ops c seed runtime)
    (h t : Nat) (hh : h<c.heads) (ht : t<c.N) :
    call.backwardInputs.gate h t=call.forwardInputs.gate h t := by
  have first := call.forwardCopies.element .gate h t 0 (Or.inr (Or.inl rfl)) ⟨hh,ht,rfl⟩
  have second := call.backwardCopies.element .gate h t 0 (Or.inr rfl) ⟨hh,ht,rfl⟩
  have inside := key_contained c seed runtime accepted ⟨.gate,h,t,0⟩ ⟨hh,ht,rfl⟩
  have saved := call.savedStorage.read .gate (Or.inr rfl) _ rfl inside.1 inside.2
  change api.read call.backwardPrepared _=_ at saved
  rw [forward_gate api ops c seed runtime accepted call.forwardPrepared h t hh ht] at saved
  exact second.symm.trans (saved.trans first)

theorem forward_caller_frame {M α β : Type} (api : API M α β) (ops : Ops α) (c : Shape) (seed : Bool)
    (runtime : Runtime c seed) (accepted : check c seed runtime=Bool.true) (call : Invocation api ops c seed runtime)
    (r : Region) (role : Preparation.ReadRole Bool.true r) (present : Checked.present c seed r=Bool.true)
    (owner address : Nat) (device : owner=(call.forwardCaller r).device)
    (lo : (call.forwardCaller r).allocationBase ≤ address)
    (hi : address<(call.forwardCaller r).allocationBase+(call.forwardCaller r).allocationBytes) :
    api.observe (forward api ops c seed runtime call.forwardPrepared).memory owner address=
      api.observe call.forwardBefore owner address := by
  rw [show api.observe (forward api ops c seed runtime call.forwardPrepared).memory owner address=
      api.observe call.forwardPrepared owner address from stage_frame api c seed runtime Bool.true _ _ _ _
      (caller_outside c seed runtime accepted _ (call.forwardLive r role present) owner address
        (by dsimp only [allocation]; exact_mod_cast lo)
        (by dsimp only [allocation]; exact_mod_cast hi) Bool.true)]
  exact call.forwardCopies.frame r role present owner address device lo hi

theorem backward_caller_frame {M α β : Type} (api : API M α β) (ops : Ops α) (c : Shape) (seed : Bool)
    (runtime : Runtime c seed) (accepted : check c seed runtime=Bool.true) (trees : Trees c)
    (call : Invocation api ops c seed runtime) (r : Region) (role : Preparation.ReadRole Bool.false r)
    (present : Checked.present c seed r=Bool.true) (owner address : Nat) (device : owner=(call.backwardCaller r).device)
    (lo : (call.backwardCaller r).allocationBase ≤ address)
    (hi : address<(call.backwardCaller r).allocationBase+(call.backwardCaller r).allocationBytes) :
    api.observe (backward api ops c seed runtime trees call.backwardPrepared).memory owner address=
      api.observe call.backwardBefore owner address := by
  have outside : StageOutside c seed runtime Bool.false owner address := by
    intro x hx cover
    have write := (stage_mem c seed Bool.false x).mp hx
    have active := active_written c seed _ _ _ _ write.1
    have inside := key_contained c seed runtime accepted x active.2
    have fresh := call.backwardFresh x.region active.1 write.2 r role present
    change ((runtime.tensor x.region).allocationBase : Int)+(runtime.tensor x.region).allocationBytes ≤
        (call.backwardCaller r).allocationBase ∨
      ((call.backwardCaller r).allocationBase : Int)+(call.backwardCaller r).allocationBytes ≤
        (runtime.tensor x.region).allocationBase at fresh
    rcases cover with ⟨_,coverLo,coverHi⟩
    rcases fresh with before|after <;> omega
  rw [show api.observe (backward api ops c seed runtime trees call.backwardPrepared).memory owner address=
      api.observe call.backwardPrepared owner address from stage_frame api c seed runtime Bool.false _ _ _ _ outside]
  exact call.backwardCopies.frame r role present owner address device lo hi

#axiomsClean forward_read
#axiomsClean saved_read
#axiomsClean backward_read
#axiomsClean native_read
#axiomsClean caller_outside
#axiomsClean invocation_saved
#axiomsClean invocation_gate
#axiomsClean forward_caller_frame
#axiomsClean backward_caller_frame
#specNonCircular SavedStorage avoiding [Source.run,Source.backwardRun,Reference.forward,Reference.reverse]
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.Host
