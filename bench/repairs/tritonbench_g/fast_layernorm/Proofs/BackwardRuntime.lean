import bench.repairs.tritonbench_g.fast_layernorm.Proofs.BackwardLayout

/-! The two actual backward launches, each bound to its own native object and
store-coordinate receipt. Values and failed-prefix frames come from source
regions; the runtime interface supplies no gradient equation. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.BackwardRuntime
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host Execution BackwardPreparation BackwardLayout Compiler
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical

inductive Kind where | dx | affine deriving DecidableEq

def Kind.entry : Kind → Compiler.Entry | .dx => .backward | .affine => .affineBackward
def Kind.count (kind : Kind) (p : Data) : Nat := match kind with | .dx => 1 | .affine => affineCount p
def Kind.output (kind : Kind) (p : Data) : Fin (kind.count p) → Dense :=
  match kind with | .dx => fun _ => p.dx | .affine => affineOutput p

def Kind.arguments (kind : Kind) (c : SavedContext) (p : Data) : List Nat :=
  match kind with
  | .dx => [c.columns,c.columns,c.columns,c.columns]
  | .affine => [c.rows,c.columns,c.w.count,c.bias.map Dense.count |>.getD 0]

def pointer (d : Dense) : Key := (d.owner,d.address 0)
def Kind.pointers (kind : Kind) (c : SavedContext) (p : Data) : List Key :=
  match kind with
  | .dx => [pointer p.dx,pointer p.dy,pointer c.x,pointer c.w,pointer c.r,pointer c.mu]
  | .affine => [pointer p.dw,pointer (p.db.getD p.dw),pointer p.dy,pointer c.x,pointer c.r,pointer c.mu]

def Kind.grid (kind : Kind) (c : SavedContext) : List Nat :=
  match kind with
  | .dx => [c.rows]
  | .affine => [(max c.w.count (c.bias.map Dense.count |>.getD 0)+127)/128]

def trace (c : SavedContext) (ops : Ops α) (plan : Plan c.columns c.block) (api : Memory.API α Store Byte)
    (p : Data) (layout : Layout c p inputs) (kind : Kind) (m : Storage.World Store) :
    Trace α (kind.count p) :=
  match kind with
  | .dx => dxTrace c ops plan api p layout m
  | .affine => affineTrace c ops plan api p layout m

theorem trace_buffer (c : SavedContext) (ops : Ops α) (plan : Plan c.columns c.block) (api : Memory.API α Store Byte)
    (p : Data) (layout : Layout c p inputs) (kind : Kind) (m : Storage.World Store)
    (k : Fin (kind.count p)) : ((trace c ops plan api p layout kind m).regions k).buffer=kind.output p k := by
  cases kind <;> rfl

theorem output_mem (kind : Kind) (p : Data) (k : Fin (kind.count p)) : kind.output p k∈outputs p := by
  cases kind with
  | dx => simp [Kind.output,outputs]
  | affine => exact affine_mem p k

structure Service (platform : Compiler.Platform) (api : Memory.API UInt32 Store UInt8)
    (neg : UInt32 → UInt32) (inputs : List Storage.Tensor) (Metadata : Type) where
  compile : (c : SavedContext) → (p : Data) → Layout c p inputs → (kind : Kind) →
    Effect api neg inputs Metadata (Compiler.Selected platform.decoder c.columns c.block)
  bound : ∀ c p layout kind s code,((compile c p layout kind).call.run s).2=.ok code →
    Compiler.Bound platform code kind.entry (kind.arguments c p) c.epsilon c.bias.isSome ∧
    code.object.signature.pointers=List.replicate 6 32
  compileOwner : ∀ c p layout kind s,((compile c p layout kind).call.run s).1.owner=s.owner
  admit : (c : SavedContext) → (p : Data) → Layout c p inputs → (kind : Kind) →
    Compiler.Selected platform.decoder c.columns c.block → Effect api neg inputs Metadata (Schedule (kind.count p))
  admitted : ∀ c p layout kind code s schedule,((admit c p layout kind code).call.run s).2=.ok schedule →
    schedule.objectIdentity=code.object.identity ∧ schedule.owner=c.x.owner ∧
    schedule.stream=s.streams c.x.owner ∧ schedule.grid=kind.grid c ∧
    (∀ entry,entry∈schedule.order → entry.2<(kind.output p entry.1).count) ∧
    (schedule.status=.ok () → ∀ k i,i<(kind.output p k).count → (k,i)∈schedule.order)
  admittedPointers : ∀ c p layout kind code s schedule,((admit c p layout kind code).call.run s).2=.ok schedule →
    schedule.pointers=kind.pointers c p
  admittedContext : ∀ c p layout kind code s schedule,((admit c p layout kind code).call.run s).2=.ok schedule →
    s.owner=c.x.owner ∧ ∀ input,input∈p.dy::([c.x,c.w,c.r,c.mu]++c.bias.toList) →
      ((admit c p layout kind code).call.run s).1.ready input.tensor.backing (s.streams c.x.owner)
  admitOwner : ∀ c p layout kind code s,((admit c p layout kind code).call.run s).1.owner=s.owner

variable {platform : Compiler.Platform} {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32}
  {inputs : List Storage.Tensor} {Metadata : Type}

def kernelMemory (c : SavedContext) (ops : Ops α) (plan : Plan c.columns c.block) (api : Memory.API α Store Byte)
    (p : Data) (layout : Layout c p inputs) (kind : Kind)
    (schedule : Schedule (kind.count p)) (m : Storage.World Store) : Storage.World Store :=
  let t := trace c ops plan api p layout kind m
  runWorld api t.owner (scheduledWrites t schedule) m

theorem kernel_frame (c : SavedContext) (ops : Ops α) (plan : Plan c.columns c.block) (api : Memory.API α Store Byte)
    (p : Data) (layout : Layout c p inputs) (kind : Kind)
    (schedule : Schedule (kind.count p)) (m : Storage.World Store) (key : Key) (hk : Storage.caller inputs key) :
    Storage.observe api (kernelMemory c ops plan api p layout kind schedule m) key=Storage.observe api m key := by
  apply trace_prefix_frame api (trace c ops plan api p layout kind m) m inputs
  · intro k old ho
    rw [trace_buffer]
    exact layout.callerFresh _ (output_mem kind p k) old ho
  · exact scheduled_subset _ schedule
  · exact hk

def kernelAction (c : SavedContext) (p : Data) (layout : Layout c p inputs) (kind : Kind)
    (code : Compiler.Selected platform.decoder c.columns c.block) (schedule : Schedule (kind.count p)) :
    Action (Host.world (Metadata:=Metadata) api inputs) Unit where
  run s :=
    let memory := kernelMemory c (platform.scalar code.object.primitives) code.plan api p layout kind schedule s.memory
    let next : State Store Metadata := {s with memory:=memory,ready:=fun backing stream => s.ready backing stream ∨
      (schedule.status=.ok () ∧ ∃ k,backing=(kind.output p k).tensor.backing ∧ stream=schedule.stream)}
    ⟨next,schedule.status.mapError (Failure.api .execute),[.execute]⟩
  frame s key hk := kernel_frame _ _ _ _ _ _ _ _ _ key hk

def dispatch (service : Service platform api neg inputs Metadata) (c : SavedContext) (p : Data)
    (layout : Layout c p inputs) (kind : Kind) : Action (Host.world (Metadata:=Metadata) api inputs) Unit :=
  (call .selectKernel (service.compile c p layout kind).call).bind fun code =>
  (call .admission (service.admit c p layout kind code).call).bind fun schedule =>
  kernelAction c p layout kind code schedule

structure Route (service : Service platform api neg inputs Metadata) (c : SavedContext) (p : Data)
    (layout : Layout c p inputs) (kind : Kind) (initial : State Store Metadata) where
  code : Compiler.Selected platform.decoder c.columns c.block
  schedule : Schedule (kind.count p)
  compiled : ((service.compile c p layout kind).call.run initial).2=.ok code
  admitted : ((service.admit c p layout kind code).call.run ((service.compile c p layout kind).call.run initial).1).2=.ok schedule
  completed : schedule.status=.ok ()

variable {service : Service platform api neg inputs Metadata} {c : SavedContext} {p : Data}
  {layout : Layout c p inputs} {kind : Kind} {initial : State Store Metadata}

def Route.s1 (r : Route service c p layout kind initial) := ((service.compile c p layout kind).call.run initial).1
def Route.s2 (r : Route service c p layout kind initial) := ((service.admit c p layout kind r.code).call.run r.s1).1
def Route.preTransition (r : Route service c p layout kind initial) : Transition api neg inputs initial r.s2 :=
  ((service.compile c p layout kind).transition initial).then
    ((service.admit c p layout kind r.code).transition r.s1)
def Route.final (r : Route service c p layout kind initial) :=
  ((kernelAction (api:=api) c p layout kind r.code r.schedule).run r.s2).state

def Route.realFinal (r : Route service c p layout kind initial) (ops : Ops ℝ)
    (m : Storage.World (Nat → ℝ)) : Storage.World (Nat → ℝ) :=
  kernelMemory c ops r.code.plan Memory.realAPI p layout kind r.schedule (r.preTransition.shadow m)

theorem dispatch_route (ok : ((dispatch service c p layout kind).run initial).result=.ok ()) :
    ∃ r : Route service c p layout kind initial,((dispatch service c p layout kind).run initial).state=r.final := by
  obtain ⟨code,hc,rest,hs,_⟩ := bind_success _ _ _ _ ok
  obtain ⟨schedule,ha,last,hs2,_⟩ := bind_success _ _ _ _ rest
  have done : schedule.status=.ok () := by
    cases h : schedule.status with
    | error e => simp [kernelAction,h,Except.mapError] at last
    | ok u => cases u; rfl
  exact ⟨⟨code,schedule,(call_ok _ _ _ _).mp hc,(call_ok _ _ _ _).mp ha,done⟩,hs.trans hs2⟩

theorem route_coverage (r : Route service c p layout kind initial) :
    ∀ k i,i<(kind.output p k).count → (k,i)∈r.schedule.order :=
  (service.admitted c p layout kind r.code r.s1 r.schedule r.admitted).2.2.2.2.2 r.completed

#axiomsClean kernel_frame
#axiomsClean dispatch_route
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.BackwardRuntime
