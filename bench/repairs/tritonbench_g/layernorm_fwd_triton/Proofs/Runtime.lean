import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.CompilerControl
import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.HostPrepared

/-! The ordinary compiler/runtime interfaces expose argument packing,
instruction decoding and grid scheduling. They contain no numerical result or
whole-kernel-output law. The actual selected object's decoded instructions are
executed below. Failed dispatch retains every completed row and any partial
output stores; caller bytes are proved preserved for those paths too. -/
namespace VeriTile.Bench.Repairs.LayernormFwdTriton
namespace Runtime
open Torch Address Memory Execution HostPrepared Compiler
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical

inductive HostScalar where
  | binary64 (bits : UInt64)
  | integer (value : Int)
  | boolean (value : Bool)

structure Platform (α : Type) where
  decoder : Decoding
  scalar : PrimitiveNames → Native.Ops α
  /-- Actual selected source scalar ABI followed by its FP32 arithmetic
  conversion. Numerical conformance is an exact local obligation, not a claim
  that every unannotated Python epsilon arrives as an FP32 ABI argument. -/
  epsilon : Object → HostScalar → α
  epsilonBinding : ScalarBinding → HostScalar → Prop
  defaultOptions : Tensor → Options

def integerBinding (binding : ScalarBinding) (value : Nat) : Prop :=
  match binding with
  | .specialized constant => constant=(value:Int)
  | .runtime .i32 => value < 2^31
  | .runtime .i64 => value < 2^63
  | .runtime .u32 => value < 2^32
  | .runtime .u64 => value < 2^64
  | _ => False

def scalarArguments (p : Data shape caller) : List Nat :=
  [p.strides.xSeq,p.strides.xHead,p.strides.xDim,p.strides.ySeq,p.strides.yHead,p.strides.yDim,
    p.strides.wHead,p.strides.wDim,shape.columns]

structure Bound (platform : Platform α) (p : Data shape caller) (eps : HostScalar)
    (code : Selected platform.decoder shape.columns) : Prop where
  slots : List.Forall₂ integerBinding code.object.signature.integerArgs (scalarArguments p)
  epsilon : platform.epsilonBinding code.object.signature.epsilon eps
  options : code.object.options=platform.defaultOptions p.x

/-- Only scheduling and exception data. Each failed-tail address is a live
output coordinate. Its supplied value models the actual completed store on an
exceptional path and never appears in a successful result theorem. -/
structure Schedule (α : Type) (shape : Shape) where
  objectIdentity : Nat
  (owner stream : Nat)
  grid : Nat × Nat
  rows : List (Row shape)
  status : Except Nat Unit
  complete : status=.ok () → ∀ row : Row shape, row∈rows
  failedTail : List (Address.Coordinate shape × α)

structure Service (platform : Platform α) (api : API α Store Byte) (neg : α → α)
    (caller : Nat → Prop) (Host : Type) where
  compile : ∀ {shape}, Data shape caller → HostScalar →
    Effect api neg caller Host (Selected platform.decoder shape.columns)
  bound : ∀ {shape} (p : Data shape caller) eps s code,
    ((compile p eps).call.run s).2=.ok code → Bound platform p eps code
  compile_owner : ∀ {shape} (p : Data shape caller) eps s, ((compile p eps).call.run s).1.owner=s.owner
  admit : ∀ {shape} (p : Data shape caller), Selected platform.decoder shape.columns →
    Effect api neg caller Host (Schedule α shape)
  admitted : ∀ {shape} (p : Data shape caller) code s schedule,
    ((admit p code).call.run s).2=.ok schedule →
    schedule.objectIdentity=code.object.identity ∧ schedule.grid=(shape.sequences,shape.heads) ∧
      schedule.owner=s.owner ∧ schedule.stream=s.streams s.owner
  admit_owner : ∀ {shape} (p : Data shape caller) code s, ((admit p code).call.run s).1.owner=s.owner

variable {α Store Byte Host : Type} {shape : Shape} {caller : Nat → Prop}

def tailWrites (p : Data shape caller) (schedule : Schedule α shape) : List (Write α) :=
  schedule.failedTail.map fun entry => ⟨yAddress p.layout entry.1.1 entry.1.2,entry.2⟩

def kernelMemory (platform : Platform α) (api : API α Store Byte) (p : Data shape caller)
    (eps : HostScalar) (code : Selected platform.decoder shape.columns)
    (schedule : Schedule α shape) (m : Store) : Store :=
  let core := CompilerControl.execute (platform.scalar code.object.primitives) code api p.layout
    (platform.epsilon code.object eps) schedule.rows m
  match schedule.status with
  | .ok _ => core
  | .error _ => Memory.run api (tailWrites p schedule) core

theorem kernel_frame (platform : Platform α) (api : API α Store Byte) (p : Data shape caller)
    (eps : HostScalar) (code : Selected platform.decoder shape.columns)
    (schedule : Schedule α shape) (m : Store) (i : Nat) (hi : caller i) :
    api.byte (kernelMemory platform api p eps code schedule m) i=api.byte m i := by
  have core : api.byte (CompilerControl.execute (platform.scalar code.object.primitives) code api p.layout
      (platform.epsilon code.object eps) schedule.rows m) i=api.byte m i := by
    rw [CompilerControl.execute_selected,Compiler.execute_grid]
    exact grid_frame _ _ api p.layout m _ schedule.rows i (HostPrepared.caller_outside p i hi)
  unfold kernelMemory
  cases h : schedule.status with
  | ok u => exact core
  | error e =>
    refine (run_byte_frame api (tailWrites p schedule) _ i ?_).trans core
    intro event he
    obtain ⟨entry,_,eq⟩ := List.mem_map.mp he
    subst event
    exact HostPrepared.caller_outside p i hi entry.1.1 entry.1.2

def kernelAction (platform : Platform α) (api : API α Store Byte) (p : Data shape caller)
    (eps : HostScalar) (code : Selected platform.decoder shape.columns)
    (schedule : Schedule α shape) : Action (Torch.world (Host:=Host) api caller) Tensor where
  run s :=
    let memory := kernelMemory platform api p eps code schedule s.memory
    let next : State Store Host := {s with memory:=memory,ready:=fun t stream => s.ready t stream ∨
      (schedule.status=.ok () ∧ t.object=p.y.object ∧ stream=schedule.stream)}
    ⟨next,(schedule.status.mapError (fun e => Failure.api .execute e)).map (fun _ => p.y),[.execute]⟩
  frame s i hi := kernel_frame platform api p eps code schedule s.memory i hi

/-- Compiler selection, actual scheduling and execution share the same
`code` value. Returning the allocated y is the last successful kernel action. -/
def dispatch (platform : Platform α) (api : API α Store Byte) (neg : α → α)
    (service : Service platform api neg caller Host) (p : Data shape caller) (eps : HostScalar) :
    Action (Torch.world (Host:=Host) api caller) Tensor :=
  (call .selectKernel (service.compile p eps).call).bind fun code =>
  (call .admission (service.admit p code).call).bind fun schedule =>
  kernelAction platform api p eps code schedule

structure Route (platform : Platform α) (api : API α Store Byte) (neg : α → α)
    (service : Service platform api neg caller Host) (p : Data shape caller) (eps : HostScalar)
    (initial : State Store Host) where
  code : Selected platform.decoder shape.columns
  schedule : Schedule α shape
  compiled : ((service.compile p eps).call.run initial).2=.ok code
  admitted : ((service.admit p code).call.run ((service.compile p eps).call.run initial).1).2=.ok schedule
  completed : schedule.status=.ok ()

variable {platform : Platform α} {api : API α Store Byte} {neg : α → α}
  {service : Service platform api neg caller Host} {p : Data shape caller} {eps : HostScalar}
  {initial : State Store Host}

def Route.s1 (r : Route platform api neg service p eps initial) : State Store Host :=
  ((service.compile p eps).call.run initial).1
def Route.s2 (r : Route platform api neg service p eps initial) : State Store Host :=
  ((service.admit p r.code).call.run r.s1).1

def Route.shadow (r : Route platform api neg service p eps initial) : (Nat → ℝ) → (Nat → ℝ) :=
  fun m => ((service.admit p r.code).shadow r.s1).run (((service.compile p eps).shadow initial).run m)

theorem dispatch_route (out : Tensor)
    (ok : ((dispatch platform api neg service p eps).run initial).result=.ok out) :
    ∃ r : Route platform api neg service p eps initial, out=p.y ∧
      ((dispatch platform api neg service p eps).run initial).state=
        ((kernelAction (Host:=Host) platform api p eps r.code r.schedule).run r.s2).state := by
  unfold dispatch at ok ⊢
  obtain ⟨code,hc,h1,s1,_⟩ := bind_success _ _ _ _ ok
  obtain ⟨schedule,ha,h2,s2,_⟩ := bind_success _ _ _ _ h1
  have completed : schedule.status=.ok () := by
    cases hh : schedule.status with
    | error e => simp [kernelAction,hh,Except.mapError,Except.map] at h2
    | ok u => cases u;rfl
  have returned : out=p.y := by
    simpa [kernelAction,completed,Except.mapError,Except.map] using h2.symm
  let r : Route platform api neg service p eps initial :=
    ⟨code,schedule,HostPrelude.call_ok _ _ _ _ hc,HostPrelude.call_ok _ _ _ _ ha,completed⟩
  exact ⟨r,returned,s1.trans s2⟩

theorem successful_memory (r : Route platform api neg service p eps initial) :
    ((kernelAction (Host:=Host) platform api p eps r.code r.schedule).run r.s2).state.memory=
      executeGrid (platform.scalar r.code.object.primitives) r.code.plan api p.layout
        (platform.epsilon r.code.object eps) r.schedule.rows r.s2.memory := by
  simp only [kernelAction,kernelMemory,r.completed]
  exact (CompilerControl.execute_selected ..).trans (Compiler.execute_grid ..)

theorem successful_readback (r : Route platform api neg service p eps initial)
    (row : Row shape) (col : Fin shape.columns) :
    api.read ((kernelAction (Host:=Host) platform api p eps r.code r.schedule).run r.s2).state.memory
        (address p.y (HostLayout.index3 row col))=
      Native.reference (platform.scalar r.code.object.primitives) r.code.plan
        (xSnapshot (platform.scalar r.code.object.primitives) api p.layout r.s2.memory row)
        (wSnapshot (platform.scalar r.code.object.primitives) api p.layout r.s2.memory row)
        (platform.epsilon r.code.object eps) col.val := by
  rw [←HostPrepared.y_address p row col,successful_memory]
  exact grid_readback _ _ api p.layout _ _ r.schedule.rows (r.schedule.complete r.completed) row col

#axiomsClean kernel_frame
#axiomsClean dispatch_route
#axiomsClean successful_memory
#axiomsClean successful_readback
#guard_msgs (drop info) in
#auditModuleAxioms
end
end Runtime
end VeriTile.Bench.Repairs.LayernormFwdTriton
