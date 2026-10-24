import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.Storage
import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.Host
open VeriTile.Triton
noncomputable section

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
  heap : Heap β

/-- Ordinary tensor-library and launch-setup outcomes. Successful allocation supplies the actual output metadata in `h` and `Placement`.
No whole-kernel or output law is part of this interface. -/
structure API (β : Type) where
  launchDevice : VeriTile.Torch.Device
  emptyLike : Heap β → Except String (Allocated β)
  setup : Nat → Config → Except String Unit

def prepare {β : Type} (api : API β) (admitted : Bool) (s : Heap β) : Except (Failure β) (Allocated β) :=
  if admitted then match api.emptyLike s with
    | .error e => .error ⟨.allocate,e,s⟩
    | .ok allocation => .ok allocation
  else .error ⟨.admission,"unsupported input",s⟩

/-- Trial invocations are ordered before the selected invocation. Every
successful kernel consumes the actual preceding heap. Setup failure returns
the heap at that exact prepare. -/
def launches {β : Type} (api : API β) (runGrid : Config → Heap β → Option (Heap β)) :
    Nat → List Config → Heap β → Except (Failure β) (Heap β)
  | _,[],s => .ok s
  | n,c::rest,s => match api.setup n c with
    | .error e => .error ⟨.setup,e,s⟩
    | .ok () => match runGrid c s with
      | none => .error ⟨.execution,"interpreter execution failed",s⟩
      | some t => launches api runGrid (n+1) rest t

/-- The API returns precisely the allocation from `empty_like`; the empty
case returns it before any autotuner/launch setup is attempted. -/
def run {β : Type} (api : API β) (runGrid : Config → Heap β → Option (Heap β))
    (h : Call) (admitted : Bool) (tried : List Config) (selected : Config) (s : Heap β) :
    Except (Failure β) (Allocated β) :=
  match prepare api admitted s with
  | .error e => .error e
  | .ok a => if h.B*h.H*h.T=0 then .ok a else
    match launches api runGrid 0 (tried++[selected]) a.heap with
    | .error e => .error e
    | .ok t => .ok {a with heap:=t}

/-- Trace of successful setup calls, including the index seen by the API. -/
def Setups {β : Type} (api : API β) : Nat → List Config → Prop
  | _,[] => True
  | n,c::rest => api.setup n c=.ok () ∧ Setups api (n+1) rest

def raw {β : Type} (runGrid : Config → Heap β → Option (Heap β)) : List Config → Heap β → Option (Heap β)
  | [],s => some s
  | c::rest,s => (runGrid c s).bind (raw runGrid rest)

theorem raw_append {β : Type} (runGrid : Config → Heap β → Option (Heap β))
    (xs ys : List Config) (s : Heap β) : raw runGrid (xs++ys) s=(raw runGrid xs s).bind (raw runGrid ys) := by
  induction xs generalizing s with
  | nil => rfl
  | cons c rest ih => simp only [List.cons_append,raw,ih,Option.bind_assoc]

theorem launches_of_setups {β : Type} (api : API β) (runGrid : Config → Heap β → Option (Heap β))
    (n : Nat) (cs : List Config) (setups : Setups api n cs) (s t : Heap β)
    (he : raw runGrid cs s=some t) : launches api runGrid n cs s=.ok t := by
  induction cs generalizing n s with
  | nil => cases he; rfl
  | cons c rest ih =>
    rcases setups with ⟨setup,setups⟩
    simp only [raw] at he
    cases runeq : runGrid c s with
    | none => simp [runeq] at he
    | some mid =>
      simp only [runeq,Option.bind_some] at he
      simp only [launches,setup,runeq]
      exact ih (n+1) setups mid he

theorem successful_return {β : Type} (api : API β) (runGrid : Config → Heap β → Option (Heap β))
    (h : Call) (admitted : Bool) (tried : List Config) (selected : Config) (s : Heap β)
    (a : Allocated β) (prepared : prepare api admitted s=.ok a) (nonempty : h.B*h.H*h.T≠0)
    (t : Heap β) (he : raw runGrid (tried++[selected]) a.heap=some t)
    (setup : Setups api 0 (tried++[selected])) :
    run api runGrid h admitted tried selected s=.ok {a with heap:=t} := by
  simp only [run,prepared,Except.bind,nonempty,if_false]
  rw [launches_of_setups api runGrid 0 _ setup _ _ he]

theorem empty_return {β : Type} (api : API β) (runGrid : Config → Heap β → Option (Heap β))
    (h : Call) (admitted : Bool) (tried : List Config) (selected : Config) (s : Heap β)
    (a : Allocated β) (prepared : prepare api admitted s=.ok a) (empty : h.B*h.H*h.T=0) :
    run api runGrid h admitted tried selected s=.ok a := by
  simp [run,prepared,empty]

/-- Failed setup preserves the exact successfully executed prepare heap. -/
theorem launches_outcome {β : Type} (api : API β) (runGrid : Config → Heap β → Option (Heap β))
    (Good : Heap β → Heap β → Prop) (refl : ∀ s, Good s s)
    (trans : ∀ a b c, Good a b → Good b c → Good a c)
    (works : ∀ c∈configurations, ∀ s, ∃ t, runGrid c s=some t ∧ Good s t)
    (n : Nat) (cs : List Config) (valid : ∀ c∈cs, c∈configurations) (s : Heap β) :
    match launches api runGrid n cs s with
    | .ok t => Good s t
    | .error e => e.stage=.setup ∧ Good s e.heap := by
  induction cs generalizing n s with
  | nil => exact refl s
  | cons c rest ih =>
    obtain ⟨mid,he,good⟩ := works c (valid c (by simp)) s
    cases setup : api.setup n c with
    | error message =>
      simpa only [launches,setup] using (show (.setup : Stage)=.setup ∧ Good s s from ⟨rfl,refl s⟩)
    | ok unit =>
      cases unit
      simp only [launches,setup,he]
      have restGood := ih (n+1) (fun k hk => valid k (List.mem_cons_of_mem c hk)) mid
      cases restEq : launches api runGrid (n+1) rest mid with
      | ok t =>
        rw [restEq] at restGood
        exact trans s mid t good restGood
      | error e =>
        rw [restEq] at restGood
        exact ⟨restGood.1,trans s mid e.heap good restGood.2⟩

def metadata (h : Call) (p : Storage.Placement h) {β : Type} (a : Allocated β) : Prop :=
  a.shape=[h.B,h.H,h.T] ∧ a.strides=h.outputStrides ∧ a.dtype=h.outputDType ∧
    a.base=p.backing.output.base ∧ a.extent=p.backing.output.extent

def preparationWrites (h : Call) (p : Storage.Placement h) (addr : Nat) : Prop :=
  (p.backing.input.base≤addr ∧ addr<p.backing.input.base+p.backing.input.extent*4) ∨
  (p.backing.output.base≤addr ∧ addr<p.backing.output.base+p.backing.output.extent*4)

/-- Direct input readback and ordinary fresh-allocation laws. No copied input,
resolution operation, kernel execution, or computed output is assumed. -/
structure PreparationLaws {β γ : Type} (api : API β) (h : Call) (p : Storage.Placement h)
    (read : β → γ) (logical : Nat → Nat → γ) (caller : Heap β) (live : Nat → Prop) : Prop where
  reads : ∀ row time, row<h.B*h.H → time<h.T →
    read (caller (Storage.address p.backing.input (h.inputStrides.address h.H row time)))=logical row time
  allocate : ∀ a, api.emptyLike caller=.ok a →
    metadata h p a ∧
    (∀ off, off<p.backing.input.extent → a.heap (Storage.address p.backing.input off)=caller (Storage.address p.backing.input off)) ∧
    (∀ addr, live addr → a.heap addr=caller addr) ∧
    (∀ addr, ¬preparationWrites h p addr → a.heap addr=caller addr)
  fresh : ∀ addr, live addr → ¬Storage.writes h p (List.finRange (h.B*h.H)) addr

theorem prepare_correct {β γ : Type} (api : API β) (h : Call) (p : Storage.Placement h)
    (read : β → γ) (logical : Nat → Nat → γ) (caller : Heap β) (live : Nat → Prop)
    (laws : PreparationLaws api h p read logical caller live) (admitted : Bool) (a : Allocated β)
    (he : prepare api admitted caller=.ok a) :
    metadata h p a ∧
    (∀ row time, row<h.B*h.H → time<h.T →
      read (a.heap (Storage.address p.backing.input (h.inputStrides.address h.H row time)))=logical row time) ∧
    (∀ addr, live addr → a.heap addr=caller addr) ∧
    (∀ addr, ¬preparationWrites h p addr → a.heap addr=caller addr) := by
  cases admitted
  · simp [prepare] at he
  · have allocated : api.emptyLike caller=.ok a := by
      cases ha : api.emptyLike caller <;> simp_all [prepare]
    obtain ⟨metadataOk,inputs,liveFrame,fullFrame⟩ := laws.allocate a allocated
    refine ⟨metadataOk,?_,liveFrame,fullFrame⟩
    intro row time hr ht
    rw [inputs _ (p.backing.inputWindow row time hr ht)]
    exact laws.reads row time hr ht

theorem output_within_preparation (h : Call) (p : Storage.Placement h) (rows : List (Fin (h.B*h.H)))
    (addr : Nat) (written : Storage.writes h p rows addr) : preparationWrites h p addr := by
  obtain ⟨row,_,time,ht,eq⟩ := written
  have bound := p.backing.outputWindow row.val time row.isLt ht
  right
  rw [eq]
  unfold Storage.address
  rw [p.outputBytes]
  omega

theorem launches_success_raw {β : Type} (api : API β) (runGrid : Config → Heap β → Option (Heap β))
    (n : Nat) (cs : List Config) (s t : Heap β) (he : launches api runGrid n cs s=.ok t) :
    raw runGrid cs s=some t := by
  induction cs generalizing n s with
  | nil => cases he; rfl
  | cons c rest ih =>
    cases setup : api.setup n c with
    | error e => simp [launches,setup] at he
    | ok unit =>
      cases unit
      cases runeq : runGrid c s with
      | none => simp [launches,setup,runeq] at he
      | some mid =>
        simp only [launches,setup,runeq] at he
        simp only [raw,runeq,Option.bind_some]
        exact ih (n+1) mid he

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

/-- Returning the allocation only replaces its heap; the device is preserved. -/
theorem returned_device {β : Type} (api : API β)
    (grid : Config → Heap β → Option (Heap β)) (h : Call) (admitted : Bool)
    (tried : List Config) (selected : Config) (caller : Heap β)
    (a result : Allocated β) (prepared : prepare api admitted caller=.ok a)
    (returned : run api grid h admitted tried selected caller=.ok result) :
    result.device=a.device := by
  simp only [run,prepared] at returned
  split at returned
  · cases returned; rfl
  · cases launched : launches api grid 0 (tried++[selected]) a.heap with
    | error e => simp [launched] at returned
    | ok final => simp [launched] at returned; cases returned; rfl

#print axioms prepare_correct

#print axioms successful_return
#print axioms empty_return
#print axioms launches_outcome
end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.Host
