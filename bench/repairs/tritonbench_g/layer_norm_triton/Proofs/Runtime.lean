import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Compiler
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Execution

namespace VeriTile.Bench.Repairs.LayerNormTriton.Runtime
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open Host Storage Execution
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical

structure Request (K : Nat) (inputs : List Storage.Tensor) where
  buffers : Buffers K inputs
  grid : List Nat
  pointers : List Key
  reads : List Dense

/-- A selected object and an ordinary launch receipt. The receipt reports only
coordinates, stream, status and readiness. It never supplies computed values.
Global grid-product/PID/segmentation compatibility remains a separate boundary. -/
structure Service (platform : Compiler.Platform) (api : Memory.API UInt32 Store UInt8)
    (neg : UInt32 → UInt32) (inputs : List Storage.Tensor) (Metadata : Type)
    (entry : Compiler.Entry) (M N B K : Nat) (eps : Compiler.Shared.HostScalar)
    (request : Request K inputs) where
  compile : Effect api neg inputs Metadata (Compiler.Selected platform.decoder entry N B)
  bound : ∀ s code,(compile.call.run s).2=.ok code → Compiler.Bound platform code M eps
  compileOwner : ∀ s,(compile.call.run s).1.owner=s.owner
  admit : Compiler.Selected platform.decoder entry N B → Effect api neg inputs Metadata (E.Schedule K)
  admitted : ∀ code s schedule,((admit code).call.run s).2=.ok schedule →
    schedule.objectIdentity=code.object.identity ∧ schedule.owner=request.buffers.owner ∧
    schedule.stream=s.streams request.buffers.owner ∧ schedule.grid=request.grid ∧ schedule.pointers=request.pointers ∧
    (∀ e,e∈schedule.order → e.2<(request.buffers.output e.1).count) ∧
    (schedule.status=.ok () → ∀ k i,i<(request.buffers.output k).count → (k,i)∈schedule.order)
  context : ∀ code s schedule,((admit code).call.run s).2=.ok schedule →
    s.owner=request.buffers.owner ∧ ∀ input,input∈request.reads →
      ((admit code).call.run s).1.ready input.tensor.backing (s.streams request.buffers.owner)
  admitOwner : ∀ code s,((admit code).call.run s).1.owner=s.owner

variable {platform : Compiler.Platform} {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32}
  {inputs : List Storage.Tensor} {Metadata : Type} {entry : Compiler.Entry} {M N B K : Nat}
  {eps : Compiler.Shared.HostScalar} {request : Request K inputs}

def execute (request : Request K inputs) (schedule : E.Schedule K)
    (values : Storage.World Store → ∀ k,Fin (request.buffers.output k).count → UInt32) :
    Action (Host.world (Metadata:=Metadata) api inputs) Unit where
  run s :=
    let memory := request.buffers.run api (values s.memory) schedule s.memory
    let next : State Store Metadata := {s with memory:=memory,ready:=fun backing stream => s.ready backing stream ∨
      (schedule.status=.ok () ∧ ∃ k,backing=(request.buffers.output k).tensor.backing ∧ stream=schedule.stream)}
    ⟨next,schedule.status.mapError (Failure.api .execute),[.execute]⟩
  frame s key hk := run_frame request.buffers api (values s.memory) schedule s.memory key hk

def dispatch (service : Service platform api neg inputs Metadata entry M N B K eps request)
    (values : Compiler.Selected platform.decoder entry N B → Storage.World Store →
      ∀ k,Fin (request.buffers.output k).count → UInt32) :
    Action (Host.world (Metadata:=Metadata) api inputs) Unit :=
  (call .selectKernel service.compile.call).bind fun code =>
  (call .admission (service.admit code).call).bind fun schedule => execute request schedule (values code)

structure Route (service : Service platform api neg inputs Metadata entry M N B K eps request)
    (values : Compiler.Selected platform.decoder entry N B → Storage.World Store →
      ∀ k,Fin (request.buffers.output k).count → UInt32) (initial : State Store Metadata) where
  code : Compiler.Selected platform.decoder entry N B
  schedule : E.Schedule K
  compiled : (service.compile.call.run initial).2=.ok code
  admitted : ((service.admit code).call.run (service.compile.call.run initial).1).2=.ok schedule
  completed : schedule.status=.ok ()

variable {service : Service platform api neg inputs Metadata entry M N B K eps request}
  {values : Compiler.Selected platform.decoder entry N B → Storage.World Store →
      ∀ k,Fin (request.buffers.output k).count → UInt32} {initial : State Store Metadata}

def Route.s1 (r : Route service values initial) := (service.compile.call.run initial).1
def Route.s2 (r : Route service values initial) := ((service.admit r.code).call.run r.s1).1

def Route.preTransition (r : Route service values initial) : Transition api neg inputs initial r.s2 :=
  (service.compile.transition initial).then ((service.admit r.code).transition r.s1)

def Route.final (r : Route service values initial) := ((execute (api:=api) request r.schedule (values r.code)).run r.s2).state

theorem route_coverage (r : Route service values initial) :
    ∀ k i,i<(request.buffers.output k).count → (k,i)∈r.schedule.order :=
  (service.admitted r.code r.s1 r.schedule r.admitted).2.2.2.2.2.2 r.completed

theorem dispatch_route (ok : ((dispatch service values).run initial).result=.ok ()) :
    ∃ r : Route service values initial,((dispatch service values).run initial).state=r.final := by
  obtain ⟨code,hc,rest,hs,_⟩ := bind_success _ _ _ _ ok
  obtain ⟨schedule,ha,last,hs2,_⟩ := bind_success _ _ _ _ rest
  have hcode := (call_ok _ _ _ _).mp hc
  have hadmit := (call_ok _ _ _ _).mp ha
  have done : schedule.status=.ok () := by
    cases h : schedule.status with
    | error e => simp [execute,h,Except.mapError] at last
    | ok u => cases u; rfl
  exact ⟨⟨code,schedule,hcode,hadmit,done⟩,hs.trans hs2⟩

theorem route_readback (r : Route service values initial) (k : Fin K) (i : Fin (request.buffers.output k).count) :
    api.read (r.final.memory request.buffers.owner) ((request.buffers.output k).address i.val)=
      values r.code r.s2.memory k i :=
  run_readback request.buffers api (values r.code r.s2.memory) r.schedule (route_coverage r) r.s2.memory k i

theorem route_owner (r : Route service values initial) : r.final.owner=initial.owner :=
  (service.admitOwner r.code r.s1).trans (service.compileOwner initial)

theorem all_outcome_frame (key : Key) (hk : Storage.caller inputs key) :
    Storage.observe api ((dispatch service values).run initial).state.memory key=Storage.observe api initial.memory key :=
  (dispatch service values).frame initial key hk

#axiomsClean dispatch_route
#axiomsClean route_readback
#axiomsClean all_outcome_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.Runtime
