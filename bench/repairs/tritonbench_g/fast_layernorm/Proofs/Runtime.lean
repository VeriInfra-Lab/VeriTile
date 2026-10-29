import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Layout

/-! Ordinary selected-object compilation and launch scheduling. The driver
receipt contains store coordinates and failure status only. Concrete source
values, byte frames and successful readback are derived below. Grid-product
compatibility/segmentation is an explicit deferred expansion. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.Runtime
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host Preparation Execution Compiler Layout
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical

structure Service (platform : Compiler.Platform) (api : Memory.API UInt32 Store UInt8)
    (neg : UInt32 → UInt32) (inputs : List Storage.Tensor) (Metadata : Type) where
  compile : ∀ {rowCount N B : Nat} (p : Data), Forward p rowCount N inputs →
    Compiler.settings N=.ok (B,Compiler.warps B) → Compiler.Shared.HostScalar →
    Effect api neg inputs Metadata (Compiler.Selected platform.decoder N B)
  bound : ∀ {rowCount N B : Nat} (p : Data) (layout : Forward p rowCount N inputs)
    (settings : Compiler.settings N=.ok (B,Compiler.warps B)) (eps : Compiler.Shared.HostScalar)
    (s : State Store Metadata) (code : Compiler.Selected platform.decoder N B),
    ((compile p layout settings eps).call.run s).2=.ok code →
      Compiler.Bound platform code .forward [N,N,N] eps p.bias.isSome ∧
      code.object.signature.pointers=List.replicate 6 32
  compileOwner : ∀ {rowCount N B : Nat} (p : Data) (layout : Forward p rowCount N inputs)
    (settings : Compiler.settings N=.ok (B,Compiler.warps B)) (eps : Compiler.Shared.HostScalar) (s : State Store Metadata),
    ((compile p layout settings eps).call.run s).1.owner=s.owner
  admit : ∀ {rowCount N B : Nat} (p : Data), Forward p rowCount N inputs →
    Compiler.Selected platform.decoder N B → Effect api neg inputs Metadata (Schedule 3)
  admitted : ∀ {rowCount N B : Nat} (p : Data) (layout : Forward p rowCount N inputs)
    (code : Compiler.Selected platform.decoder N B) (s : State Store Metadata) (schedule : Schedule 3),
    ((admit p layout code).call.run s).2=.ok schedule →
      schedule.objectIdentity=code.object.identity ∧ schedule.owner=p.x.owner ∧
      schedule.stream=s.streams p.x.owner ∧ schedule.grid=[rowCount] ∧
      (∀ entry,entry∈schedule.order → entry.2<(outputs p entry.1).count) ∧
      (schedule.status=.ok () → ∀ k i,i<(outputs p k).count → (k,i)∈schedule.order)
  admittedPointers : ∀ {rowCount N B : Nat} (p : Data) (layout : Forward p rowCount N inputs)
    (code : Compiler.Selected platform.decoder N B) (s : State Store Metadata) (schedule : Schedule 3),
    ((admit p layout code).call.run s).2=.ok schedule →
      schedule.pointers=[(p.y.owner,p.y.address 0),(p.x.owner,p.x.address 0),(p.w.owner,p.w.address 0),
        ((p.bias.getD p.w).owner,(p.bias.getD p.w).address 0),(p.r.owner,p.r.address 0),(p.mu.owner,p.mu.address 0)]
  admittedContext : ∀ {rowCount N B : Nat} (p : Data) (layout : Forward p rowCount N inputs)
    (code : Compiler.Selected platform.decoder N B) (s : State Store Metadata) (schedule : Schedule 3),
    ((admit p layout code).call.run s).2=.ok schedule → s.owner=p.x.owner ∧
      ∀ input,input∈reads p → ((admit p layout code).call.run s).1.ready input.tensor.backing (s.streams p.x.owner)
  admitOwner : ∀ {rowCount N B : Nat} (p : Data) (layout : Forward p rowCount N inputs)
    (code : Compiler.Selected platform.decoder N B) (s : State Store Metadata),
    ((admit p layout code).call.run s).1.owner=s.owner

variable {platform : Compiler.Platform} {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32}
  {inputs : List Storage.Tensor} {Metadata : Type} {rowCount N B : Nat}

def forwardMemory (ops : Ops α) (plan : Plan N B) (api : Memory.API α Store Byte)
    (p : Data) (layout : Forward p rowCount N inputs) (eps : α)
    (schedule : Schedule 3) (m : Storage.World Store) : Storage.World Store :=
  let t := forwardTrace ops plan api p layout eps m
  runWorld api t.owner (scheduledWrites t schedule) m

theorem forward_frame (ops : Ops α) (plan : Plan N B) (api : Memory.API α Store Byte)
    (p : Data) (layout : Forward p rowCount N inputs) (eps : α)
    (schedule : Schedule 3) (m : Storage.World Store) (key : Key) (hk : Storage.caller inputs key) :
    Storage.observe api (forwardMemory ops plan api p layout eps schedule m) key=Storage.observe api m key := by
  apply trace_prefix_frame api (forwardTrace ops plan api p layout eps m) m inputs
  · intro k old ho
    have hf := layout.callerFresh k old ho
    cases hb : p.bias <;> fin_cases k <;>
      simpa [forwardTrace,hb,forwardRegions,forwardRegionsNoBias,outputs] using hf
  · exact scheduled_subset _ schedule
  · exact hk

def kernelAction (p : Data) (layout : Forward p rowCount N inputs)
    (code : Compiler.Selected platform.decoder N B) (eps : Compiler.Shared.HostScalar)
    (schedule : Schedule 3) : Action (Host.world (Metadata:=Metadata) api inputs) Data where
  run s :=
    let memory := forwardMemory (platform.scalar code.object.primitives) code.plan api p layout
      (platform.epsilon code.object eps) schedule s.memory
    let next : State Store Metadata := {s with memory:=memory,ready:=fun backing stream => s.ready backing stream ∨
      (schedule.status=.ok () ∧ ∃ k,backing=(outputs p k).tensor.backing ∧ stream=schedule.stream)}
    ⟨next,(schedule.status.mapError (Failure.api .execute)).map (fun _ => p),[.execute]⟩
  frame s key hk := forward_frame _ _ _ _ _ _ _ _ key hk

def dispatch (service : Service platform api neg inputs Metadata) (p : Data)
    (layout : Forward p rowCount N inputs) (settings : Compiler.settings N=.ok (B,Compiler.warps B))
    (eps : Compiler.Shared.HostScalar) : Action (Host.world (Metadata:=Metadata) api inputs) Data :=
  (call .selectKernel (service.compile p layout settings eps).call).bind fun code =>
  (call .admission (service.admit p layout code).call).bind fun schedule =>
  kernelAction p layout code eps schedule

structure Route (service : Service platform api neg inputs Metadata) (p : Data)
    (layout : Forward p rowCount N inputs) (settings : Compiler.settings N=.ok (B,Compiler.warps B))
    (eps : Compiler.Shared.HostScalar) (initial : State Store Metadata) where
  code : Compiler.Selected platform.decoder N B
  schedule : Schedule 3
  compiled : ((service.compile p layout settings eps).call.run initial).2=.ok code
  admitted : ((service.admit p layout code).call.run ((service.compile p layout settings eps).call.run initial).1).2=.ok schedule
  completed : schedule.status=.ok ()

variable {service : Service platform api neg inputs Metadata} {p : Data}
  {layout : Forward p rowCount N inputs} {settings : Compiler.settings N=.ok (B,Compiler.warps B)}
  {eps : Compiler.Shared.HostScalar} {initial : State Store Metadata}

def Route.s1 (r : Route service p layout settings eps initial) :=
  ((service.compile p layout settings eps).call.run initial).1
def Route.s2 (r : Route service p layout settings eps initial) :=
  ((service.admit p layout r.code).call.run r.s1).1
def Route.preTransition (r : Route service p layout settings eps initial) : Transition api neg inputs initial r.s2 :=
  ((service.compile p layout settings eps).transition initial).then
    ((service.admit p layout r.code).transition r.s1)
def Route.final (r : Route service p layout settings eps initial) :=
  ((kernelAction (api:=api) p layout r.code eps r.schedule).run r.s2).state

/-- Mathematical shadow of the same compiled object, allocations and schedule.
The caller supplies arbitrary Real memory and epsilon after fixing the native
route; no successful second Real dispatch and no FP decoding are involved. -/
def Route.realFinal (r : Route service p layout settings eps initial) (ops : Ops ℝ)
    (m : Storage.World (Nat → ℝ)) (epsReal : ℝ) : Storage.World (Nat → ℝ) :=
  forwardMemory ops r.code.plan Memory.realAPI p layout epsReal r.schedule (r.preTransition.shadow m)

theorem dispatch_route (out : Data)
    (ok : ((dispatch service p layout settings eps).run initial).result=.ok out) :
    ∃ r : Route service p layout settings eps initial,out=p ∧
      ((dispatch service p layout settings eps).run initial).state=r.final := by
  obtain ⟨code,hc,rest,hs,_⟩ := bind_success _ _ _ _ ok
  obtain ⟨schedule,ha,last,hs2,_⟩ := bind_success _ _ _ _ rest
  have hcode := (call_ok _ _ _ _).mp hc
  have hadmit := (call_ok _ _ _ _).mp ha
  have done : schedule.status=.ok () := by
    cases h : schedule.status with
    | error e => simp [kernelAction,h,Except.mapError,Except.map] at last
    | ok u => cases u; rfl
  have hout : out=p := by simpa [kernelAction,done,Except.mapError,Except.map] using last.symm
  exact ⟨⟨code,schedule,hcode,hadmit,done⟩,hout,hs.trans hs2⟩

#axiomsClean forward_frame
#axiomsClean dispatch_route
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.Runtime
