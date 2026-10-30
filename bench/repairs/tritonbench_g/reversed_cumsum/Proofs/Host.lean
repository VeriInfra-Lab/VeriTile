import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.Storage
import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.Host

namespace VeriTile.Bench.Repairs.ReversedCumsum.Host
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3600000
set_option maxRecDepth 24000

abbrev Heap (β : Type) := Nat → β
abbrev Allocated := ReversedCumsumScalar.Host.Allocated
abbrev Stage := ReversedCumsumScalar.Host.Stage
abbrev Failure := ReversedCumsumScalar.Host.Failure
abbrev API := ReversedCumsumScalar.Host.API
abbrev launchAPI {β : Type} (api : API β) := api
abbrev failure {β : Type} (e : Failure β) := e
abbrev prepare := @ReversedCumsumScalar.Host.prepare

def Primary (h : Call) : Prop := h.inputDType=.fp32 ∧ h.outputDType=.fp32

def run {β : Type} (api : API β) (runGrid : Config → Heap β → Option (Heap β))
    (h : Call) (admitted : Bool) (tried : List Config) (selected : Config) (s : Heap β) : Except (Failure β) (Allocated β) :=
  match prepare api admitted s with
  | .error e => .error e
  | .ok a => if h.B*h.H*(h.T*h.S)=0 then .ok a else
    match ReversedCumsumScalar.Host.launches (launchAPI api) runGrid 0 (tried++[selected]) a.heap with
    | .error e => .error (failure e)
    | .ok t => .ok {a with heap:=t}

/-- Canonical active-coordinate projection, including arbitrary singleton strides. -/
def metadata (h : Call) (p : Storage.Placement h) {β : Type} (a : Allocated β) : Prop :=
  a.shape=[h.B,h.H,h.T,h.S] ∧ a.strides=h.flat.outputStrides ∧ a.dtype=h.outputDType ∧
    a.base=p.backing.output.base ∧ a.extent=p.backing.output.extent

def preparationWrites (h : Call) (p : Storage.Placement h) (addr : Nat) : Prop :=
  ReversedCumsumScalar.Host.preparationWrites h.flat p addr

/-- Original contiguous input reads and ordinary fresh empty_like allocation.
There is no input normalization, numerical output, or execution premise. -/
structure PreparationLaws {β γ : Type} (api : API β) (h : Call) (p : Storage.Placement h)
    (read : β → γ) (logical : Nat → Nat → Nat → γ)
    (caller : Heap β) (live : Nat → Prop) : Prop where
  reads : ∀ row t s, row<h.B*h.H → t<h.T → s<h.S →
    read (caller (Storage.address p.backing.input (Geometry.address h row t s)))=logical row t s
  allocate : ∀ a, api.emptyLike caller=.ok a →
    metadata h p a ∧
    (∀ off, off<p.backing.input.extent → a.heap (Storage.address p.backing.input off)=caller (Storage.address p.backing.input off)) ∧
    (∀ addr, live addr → a.heap addr=caller addr) ∧
    (∀ addr, ¬preparationWrites h p addr → a.heap addr=caller addr)
  fresh : ∀ addr, live addr → ¬Storage.writes h p (Storage.programs h) addr

theorem prepare_correct {β γ : Type} (api : API β) (h : Call) (p : Storage.Placement h)
    (read : β → γ) (logical : Nat → Nat → Nat → γ)
    (caller : Heap β) (live : Nat → Prop)
    (laws : PreparationLaws api h p read logical caller live) (admitted : Bool) (a : Allocated β)
    (he : prepare api admitted caller=.ok a) :
    metadata h p a ∧
    (∀ row t s, row<h.B*h.H → t<h.T → s<h.S →
      read (a.heap (Storage.address p.backing.input (Geometry.address h row t s)))=logical row t s) ∧
    (∀ addr, live addr → a.heap addr=caller addr) ∧
    (∀ addr, ¬preparationWrites h p addr → a.heap addr=caller addr) := by
  cases admitted
  · simp [prepare,ReversedCumsumScalar.Host.prepare] at he
  · have allocated : api.emptyLike caller=.ok a := by
      cases ha : api.emptyLike caller <;> simp_all [prepare,ReversedCumsumScalar.Host.prepare]
    obtain ⟨metadataOk,inputs,liveFrame,fullFrame⟩ := laws.allocate a allocated
    refine ⟨metadataOk,?_,liveFrame,fullFrame⟩
    intro row t s hr ht hs
    rw [inputs _ (Storage.window h p ⟨row,hr⟩ t s ht hs).1]
    exact laws.reads row t s hr ht hs

abbrev prepare_failure := @ReversedCumsumScalar.Host.prepare_failure

theorem successful_return {β : Type} (api : API β) (runGrid : Config → Heap β → Option (Heap β))
    (h : Call) (rank4 : Bool) (tried : List Config) (selected : Config) (s : Heap β)
    (a : Allocated β) (prepared : prepare api rank4 s=.ok a) (nonempty : h.B*h.H*(h.T*h.S)≠0)
    (t : Heap β) (he : ReversedCumsumScalar.Host.raw runGrid (tried++[selected]) a.heap=some t)
    (setup : ReversedCumsumScalar.Host.Setups (launchAPI api) 0 (tried++[selected])) :
    run api runGrid h rank4 tried selected s=.ok {a with heap:=t} := by
  simp only [run,prepared,nonempty,if_false]
  rw [ReversedCumsumScalar.Host.launches_of_setups (launchAPI api) runGrid 0 _ setup _ _ he]

theorem empty_return {β : Type} (api : API β) (runGrid : Config → Heap β → Option (Heap β))
    (h : Call) (rank4 : Bool) (tried : List Config) (selected : Config) (s : Heap β)
    (a : Allocated β) (prepared : prepare api rank4 s=.ok a) (empty : h.B*h.H*(h.T*h.S)=0) :
    run api runGrid h rank4 tried selected s=.ok a := by simp [run,prepared,empty]

/-- Generic protocol composition, instantiated publicly with the proved grid
executions. Only setup can fail after successful preparation. -/
theorem launches_outcome {β : Type} (api : API β) (runGrid : Config → Heap β → Option (Heap β))
    (Good : Heap β → Heap β → Prop) (refl : ∀ s,Good s s)
    (trans : ∀ a b c,Good a b → Good b c → Good a c)
    (n : Nat) (cs : List Config)
    (works : ∀ c∈cs,∀ s,∃ t,runGrid c s=some t ∧ Good s t) (s : Heap β) :
    match ReversedCumsumScalar.Host.launches (launchAPI api) runGrid n cs s with
    | .ok t => Good s t
    | .error e => e.stage=.setup ∧ Good s e.heap := by
  induction cs generalizing n s with
  | nil => exact refl s
  | cons c rest ih =>
    obtain ⟨mid,he,good⟩ := works c (by simp) s
    cases setup : api.setup n c with
    | error msg =>
      simpa only [ReversedCumsumScalar.Host.launches,launchAPI,setup] using
        (show (ReversedCumsumScalar.Host.Stage.setup)=.setup ∧ Good s s from ⟨rfl,refl s⟩)
    | ok unit =>
      cases unit
      simp only [ReversedCumsumScalar.Host.launches,launchAPI,setup,he]
      have restGood := ih (n+1) (fun k hk => works k (List.mem_cons_of_mem c hk)) mid
      cases restEq : ReversedCumsumScalar.Host.launches (launchAPI api) runGrid (n+1) rest mid with
      | ok t =>
        rw [restEq] at restGood
        exact trans s mid t good restGood
      | error e =>
        rw [restEq] at restGood
        exact ⟨restGood.1,trans s mid e.heap good restGood.2⟩

theorem returned_device {β : Type} (api : API β)
    (grid : Config → Heap β → Option (Heap β)) (h : Call) (admitted : Bool)
    (tried : List Config) (selected : Config) (caller : Heap β)
    (a result : Allocated β) (prepared : prepare api admitted caller=.ok a)
    (returned : run api grid h admitted tried selected caller=.ok result) :
    result.device=a.device := by
  simp only [run,prepared] at returned
  split at returned
  · cases returned; rfl
  · cases launched : ReversedCumsumScalar.Host.launches api grid 0 (tried++[selected]) a.heap with
    | error e => simp [launchAPI,launched] at returned
    | ok final => simp [launchAPI,launched] at returned; cases returned; rfl

#axiomsClean prepare_correct
#axiomsClean returned_device
end
end VeriTile.Bench.Repairs.ReversedCumsum.Host
