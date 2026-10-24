import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.Stream
import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.Host
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
abbrev Heap (β : Type) := Nat → β

def Primary (h : Call) : Prop := h.inputDType=.fp32 ∧ h.outputDType=.fp32
inductive Stage where
  | admission | allocate | setup | execution
  deriving DecidableEq
structure Failure (β : Type) where
  stage : Stage
  message : String
  heap : Heap β
structure Allocated (β : Type) where
  identity : Nat
  device : VeriTile.Torch.Device
  shape : List Nat
  strides : Strides
  dtype : FloatDType
  base : Nat
  extent : Nat
  bytes : Nat
  storageOffset : Nat
  heap : Heap β
structure API (β : Type) where
  launchDevice : VeriTile.Torch.Device
  emptyLike : Heap β → Except String (Allocated β)
  setup : Nat → (Nat × Nat) → Config → Except String Unit

def prepare {β : Type} (api : API β) (admitted : Bool) (s : Heap β) : Except (Failure β) (Allocated β) :=
  if admitted then match api.emptyLike s with
    | .error e => .error ⟨.allocate,e,s⟩
    | .ok allocation => .ok allocation
  else .error ⟨.admission,"unsupported input",s⟩

/-- Each setup and execution consumes its actual preceding heap. The ordinal
also permits a fresh arbitrary padding oracle on every trial and selected run. -/
def launches {β : Type} (api : API β) (h : Call)
    (runGrid : Nat → Config → Heap β → Option (Heap β)) :
    Nat → List Config → Heap β → Except (Failure β) (Heap β)
  | _,[],s => .ok s
  | n,c::rest,s => match api.setup n (Geometry.features h,h.B*h.H) c with
    | .error e => .error ⟨.setup,e,s⟩
    | .ok () => match runGrid n c s with
      | none => .error ⟨.execution,"interpreter execution failed",s⟩
      | some t => launches api h runGrid (n+1) rest t

def run {β : Type} (api : API β) (runGrid : Nat → Config → Heap β → Option (Heap β))
    (h : Call) (admitted : Bool) (tried : List Config) (selected : Config) (s : Heap β) :
    Except (Failure β) (Allocated β) :=
  match prepare api admitted s with
  | .error e => .error e
  | .ok a => if a.shape.prod=0 then .ok a else
    match launches api h runGrid 0 (tried++[selected]) a.heap with
    | .error e => .error e
    | .ok t => .ok {a with heap:=t}

def Setups {β : Type} (api : API β) (h : Call) : Nat → List Config → Prop
  | _,[] => True
  | n,c::rest => api.setup n (Geometry.features h,h.B*h.H) c=.ok () ∧ Setups api h (n+1) rest

def raw {β : Type} (runGrid : Nat → Config → Heap β → Option (Heap β)) : Nat → List Config → Heap β → Option (Heap β)
  | _,[],s => some s
  | n,c::rest,s => (runGrid n c s).bind (raw runGrid (n+1) rest)

theorem raw_append {β : Type} (runGrid : Nat → Config → Heap β → Option (Heap β))
    (n : Nat) (xs ys : List Config) (s : Heap β) :
    raw runGrid n (xs++ys) s=(raw runGrid n xs s).bind (raw runGrid (n+xs.length) ys) := by
  induction xs generalizing n s with
  | nil => simp [raw]
  | cons c rest ih =>
    simp only [List.cons_append,raw,ih,Option.bind_assoc,List.length_cons,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

theorem launches_of_setups {β : Type} (api : API β) (h : Call)
    (runGrid : Nat → Config → Heap β → Option (Heap β))
    (n : Nat) (cs : List Config) (setups : Setups api h n cs) (s t : Heap β)
    (he : raw runGrid n cs s=some t) : launches api h runGrid n cs s=.ok t := by
  induction cs generalizing n s with
  | nil => cases he; rfl
  | cons c rest ih =>
    rcases setups with ⟨setup,setups⟩
    simp only [raw] at he
    cases runeq : runGrid n c s with
    | none => simp [runeq] at he
    | some mid =>
      simp only [runeq,Option.bind_some] at he
      simp only [launches,setup,runeq]
      exact ih (n+1) setups mid he

theorem successful_return {β : Type} (api : API β)
    (runGrid : Nat → Config → Heap β → Option (Heap β))
    (h : Call) (admitted : Bool) (tried : List Config) (selected : Config) (s : Heap β)
    (a : Allocated β) (prepared : prepare api admitted s=.ok a) (nonempty : a.shape.prod≠0)
    (t : Heap β) (he : raw runGrid 0 (tried++[selected]) a.heap=some t)
    (setup : Setups api h 0 (tried++[selected])) :
    run api runGrid h admitted tried selected s=.ok {a with heap:=t} := by
  simp only [run,prepared,nonempty,if_false]
  rw [launches_of_setups api h runGrid 0 _ setup _ _ he]

theorem empty_return {β : Type} (api : API β)
    (runGrid : Nat → Config → Heap β → Option (Heap β))
    (h : Call) (admitted : Bool) (tried : List Config) (selected : Config) (s : Heap β)
    (a : Allocated β) (prepared : prepare api admitted s=.ok a) (empty : a.shape.prod=0) :
    run api runGrid h admitted tried selected s=.ok a := by simp [run,prepared,empty]

theorem launches_outcome {β : Type} (api : API β) (h : Call)
    (runGrid : Nat → Config → Heap β → Option (Heap β))
    (Good : Heap β → Heap β → Prop) (refl : ∀ s, Good s s)
    (trans : ∀ a b c, Good a b → Good b c → Good a c)
    (works : ∀ n c, c∈configurations → ∀ s, ∃ t, runGrid n c s=some t ∧ Good s t)
    (n : Nat) (cs : List Config) (valid : ∀ c∈cs, c∈configurations) (s : Heap β) :
    match launches api h runGrid n cs s with
    | .ok t => Good s t
    | .error e => e.stage=.setup ∧ Good s e.heap := by
  induction cs generalizing n s with
  | nil => exact refl s
  | cons c rest ih =>
    obtain ⟨mid,he,good⟩ := works n c (valid c (by simp)) s
    cases setup : api.setup n (Geometry.features h,h.B*h.H) c with
    | error message =>
      simpa only [launches,setup] using (show (.setup : Stage)=.setup ∧ Good s s from ⟨rfl,refl s⟩)
    | ok unit =>
      cases unit
      simp only [launches,setup,he]
      have restGood := ih (n+1) (fun k hk => valid k (List.mem_cons_of_mem c hk)) mid
      cases restEq : launches api h runGrid (n+1) rest mid with
      | ok t =>
        rw [restEq] at restGood
        exact trans s mid t good restGood
      | error e =>
        rw [restEq] at restGood
        exact ⟨restGood.1,trans s mid e.heap good restGood.2⟩

def metadata (h : Call) (p : Storage.Placement h) {β : Type} (a : Allocated β) : Prop :=
  a.shape=[h.B,h.H,h.T,h.S] ∧ a.strides=h.outputStrides ∧ a.dtype=h.outputDType ∧
    a.base=p.backing.output.base ∧ a.extent=p.backing.output.extent ∧ a.bytes=p.backing.output.bytes ∧ a.storageOffset=0

def preparationWrites (h : Call) (p : Storage.Placement h) (addr : Nat) : Prop :=
  (p.backing.input.base≤addr ∧ addr<p.backing.input.base+p.backing.input.extent*p.backing.input.bytes) ∨
  (p.backing.output.base≤addr ∧ addr<p.backing.output.base+p.backing.output.extent*p.backing.output.bytes)

/-- Direct input readback and ordinary fresh-allocation laws. No copied input,
resolution operation, kernel execution, or computed output is assumed. -/
structure PreparationLaws {β γ : Type} (api : API β) (h : Call) (p : Storage.Placement h)
    (read : β → γ) (logical : Nat → Nat → Nat → γ) (caller : Heap β) (live : Nat → Prop) : Prop where
  reads : ∀ row time channel, row<h.B*h.H → time<h.T → channel<h.S →
    read (caller (Storage.address p.backing.input (h.inputStrides.address h.H row time channel)))=logical row time channel
  allocate : ∀ a, api.emptyLike caller=.ok a →
    metadata h p a ∧
    (∀ off, off<p.backing.input.extent → a.heap (Storage.address p.backing.input off)=caller (Storage.address p.backing.input off)) ∧
    (∀ addr, live addr → a.heap addr=caller addr) ∧
    (∀ addr, ¬preparationWrites h p addr → a.heap addr=caller addr)
  fresh : ∀ addr, live addr → ¬Storage.writes h p (Grid.programs h) addr

theorem prepare_correct {β γ : Type} (api : API β) (h : Call) (p : Storage.Placement h)
    (read : β → γ) (logical : Nat → Nat → Nat → γ) (caller : Heap β) (live : Nat → Prop)
    (laws : PreparationLaws api h p read logical caller live) (admitted : Bool) (a : Allocated β)
    (he : prepare api admitted caller=.ok a) :
    metadata h p a ∧
    (∀ row time channel, row<h.B*h.H → time<h.T → channel<h.S →
      read (a.heap (Storage.address p.backing.input (h.inputStrides.address h.H row time channel)))=logical row time channel) ∧
    (∀ addr, live addr → a.heap addr=caller addr) ∧
    (∀ addr, ¬preparationWrites h p addr → a.heap addr=caller addr) := by
  cases admitted
  · simp [prepare] at he
  · have allocated : api.emptyLike caller=.ok a := by
      cases ha : api.emptyLike caller <;> simp_all [prepare]
    obtain ⟨metadataOk,inputs,liveFrame,fullFrame⟩ := laws.allocate a allocated
    refine ⟨metadataOk,?_,liveFrame,fullFrame⟩
    intro row time channel hr ht hc
    rw [inputs _ (p.backing.inputWindow row time channel hr ht hc)]
    exact laws.reads row time channel hr ht hc

theorem output_within_preparation (h : Call) (p : Storage.Placement h)
    (programs : List (Geometry.Program h)) (addr : Nat) (written : Storage.writes h p programs addr) :
    preparationWrites h p addr := by
  obtain ⟨prog,_,time,j,ha,eq⟩ := written
  have bound := p.backing.outputWindow prog.row time (prog.feature*BS+j.val) prog.rowInside ha.1 ha.2
  have bytes := p.backing.output.bytesPositive
  have offset := Nat.mul_lt_mul_of_pos_right bound bytes
  right
  rw [eq]
  unfold Storage.address Grid.offset
  omega

def Good {β : Type} (h : Call) (p : Storage.Placement h) (s t : Heap β) : Prop :=
  Stream.StableInput h p s t ∧ (∀ addr, ¬Storage.writes h p (Grid.programs h) addr → t addr=s addr)

theorem good_refl {β : Type} (h : Call) (p : Storage.Placement h) (s : Heap β) : Good h p s s :=
  ⟨fun _ _ => rfl,fun _ _ => rfl⟩
theorem good_trans {β : Type} (h : Call) (p : Storage.Placement h) (a b c : Heap β)
    (ab : Good h p a b) (bc : Good h p b c) : Good h p a c :=
  ⟨fun off hi => (bc.1 off hi).trans (ab.1 off hi),fun addr hn => (bc.2 addr hn).trans (ab.2 addr hn)⟩

theorem raw_good {β : Type} (h : Call) (p : Storage.Placement h)
    (runGrid : Nat → Config → Heap β → Option (Heap β))
    (works : ∀ n c, c∈configurations → ∀ s, ∃ t, runGrid n c s=some t ∧ Good h p s t)
    (n : Nat) (cs : List Config) (valid : ∀ c∈cs, c∈configurations) (s : Heap β) :
    ∃ t, raw runGrid n cs s=some t ∧ Good h p s t := by
  induction cs generalizing n s with
  | nil => exact ⟨s,rfl,good_refl h p s⟩
  | cons c rest ih =>
    obtain ⟨mid,he,good⟩ := works n c (valid c (by simp)) s
    obtain ⟨t,ht,good'⟩ := ih (n+1) (fun k hk => valid k (List.mem_cons_of_mem c hk)) mid
    exact ⟨t,by simp only [raw,he,Option.bind_some,ht],good_trans h p s mid t good good'⟩

theorem selected_complete {β γ : Type} (h : Call) (p : Storage.Placement h)
    (read : β → γ) (runGrid : Nat → Config → Heap β → Option (Heap β))
    (expected : Nat → Config → Heap β → Geometry.Program h → Nat → Fin BS → γ)
    (works : ∀ n c, c∈configurations → ∀ s, ∃ t, runGrid n c s=some t ∧ Good h p s t ∧
      (∀ prog time j, Grid.active h prog time j →
        read (t (Storage.address p.backing.output (Grid.offset h prog time j)))=expected n c s prog time j))
    (stable : ∀ n c, c∈configurations → ∀ a b, Stream.StableInput h p a b → ∀ prog time j,
      Grid.active h prog time j → expected n c b prog time j=expected n c a prog time j)
    (tried : List Config) (valid : ∀ c∈tried, c∈configurations)
    (selected : Config) (chosen : selected∈configurations) (s : Heap β) :
    ∃ t, raw runGrid 0 (tried++[selected]) s=some t ∧ Good h p s t ∧
      (∀ prog time j, Grid.active h prog time j →
        read (t (Storage.address p.backing.output (Grid.offset h prog time j)))=expected tried.length selected s prog time j) := by
  obtain ⟨mid,he,good⟩ := raw_good h p runGrid
    (fun n c hc s => by obtain ⟨t,he,hg,_⟩ := works n c hc s; exact ⟨t,he,hg⟩) 0 tried valid s
  obtain ⟨t,ht,good',out⟩ := works tried.length selected chosen mid
  refine ⟨t,?_,good_trans h p s mid t good good',?_⟩
  · rw [raw_append,he]
    simp only [Nat.zero_add,Option.bind_some,raw,ht]
  · intro prog time j ha
    exact (out prog time j ha).trans (stable tried.length selected chosen s mid good.1 prog time j ha)

theorem prepare_failure {β : Type} (api : API β) (admitted : Bool)
    (caller : Heap β) (e : Failure β) (he : prepare api admitted caller=.error e) :
    (e.stage=.admission ∨ e.stage=.allocate) ∧ e.heap=caller := by
  cases admitted
  · simp [prepare] at he
    cases he
    exact ⟨Or.inl rfl,rfl⟩
  · cases ha : api.emptyLike caller with
    | ok a => simp [prepare,ha] at he
    | error message =>
      simp [prepare,ha] at he
      cases he
      exact ⟨Or.inr rfl,rfl⟩

#axiomsClean selected_complete
#axiomsClean prepare_failure
#axiomsClean prepare_correct
#axiomsClean successful_return
#axiomsClean launches_outcome
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.Host
