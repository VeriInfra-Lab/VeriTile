import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.RealRevision.RealMemory

/-! Same actual local operations with the arbitrary Real caller-cell frame.
These per-call contracts include protected backing cells and padding, beyond
logical coordinates of currently live tensor views. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.RealRevision
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper WrapperNative
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000
variable {α Host : Type} {caller : Address → Prop}

def physicalWorld (ρ : HostState α Host → Address → ℝ) (caller : Address → Prop) :
    World (HostState α Host) Address MemCell := ⟨interpret ρ,caller⟩

def LocalFrame (ρ : HostState α Host → Address → ℝ) {β : Type}
    (operation : Local (world (α:=α) (Host:=Host) caller) β) : Prop :=
  ∀ s i, caller i → interpret ρ (operation.run s).1 i=interpret ρ s i

structure TensorFrames (ρ : HostState α Host → Address → ℝ)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller)) : Prop where
  resolve : ∀ role input, LocalFrame ρ (tensor.resolve role input)
  transfer : ∀ target role input, LocalFrame ρ (tensor.transfer target role input)
  allocate : ∀ role request, LocalFrame ρ (tensor.allocate role request)
  power : ∀ n, LocalFrame ρ (tensor.nextPower n)

structure DispatchFrames (ρ : HostState α Host → Address → ℝ)
    (device : DispatchAPI (world (α:=α) (Host:=Host) caller)) : Prop where
  enter : ∀ n, LocalFrame ρ (device.enter n)
  leave : ∀ n, LocalFrame ρ (device.leave n)
  select : ∀ n, LocalFrame ρ (device.select n)
  admit : ∀ n, LocalFrame ρ (device.admit n)
  split : ∀ n, LocalFrame ρ (device.split n)

def framedLocal (ρ : HostState α Host → Address → ℝ) {β : Type}
    (operation : Local (world (α:=α) (Host:=Host) caller) β) (h : LocalFrame ρ operation) :
    Local (physicalWorld ρ caller) β := ⟨operation.run,h⟩

def framedTensor (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (ρ : HostState α Host → Address → ℝ) (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (laws : Transport store ρ tensor) (frames : TensorFrames ρ tensor) : TensorAPI (physicalWorld ρ caller) where
  read := (shadowStore store ρ).read
  callerObjects := tensor.callerObjects
  resolve role input := framedLocal ρ (tensor.resolve role input) (frames.resolve role input)
  resolve_metadata := tensor.resolve_metadata
  resolve_values := laws.resolve
  transfer target role input := framedLocal ρ (tensor.transfer target role input) (frames.transfer target role input)
  transfer_metadata := tensor.transfer_metadata
  transfer_values := laws.transfer
  allocate role request := framedLocal ρ (tensor.allocate role request) (frames.allocate role request)
  allocate_metadata := tensor.allocate_metadata
  nextPower n := framedLocal ρ (tensor.nextPower n) (frames.power n)
  nextPower_value := tensor.nextPower_value

def framedDispatch (ρ : HostState α Host → Address → ℝ)
    (device : DispatchAPI (world (α:=α) (Host:=Host) caller)) (frames : DispatchFrames ρ device) :
    DispatchAPI (physicalWorld ρ caller) where
  enter n := framedLocal ρ (device.enter n) (frames.enter n)
  leave n := framedLocal ρ (device.leave n) (frames.leave n)
  select n := framedLocal ρ (device.select n) (frames.select n)
  admit n := framedLocal ρ (device.admit n) (frames.admit n)
  split n := framedLocal ρ (device.split n) (frames.split n)

theorem framed_prepare_run
    (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (ρ : HostState α Host → Address → ℝ) (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (laws : Transport store ρ tensor) (frames : TensorFrames ρ tensor) (q : Request) :
    (prepare (framedTensor store ρ tensor laws frames) q).run=(prepare tensor q).run := by
  let ft := framedTensor store ρ tensor laws frames
  have opt (target : Nat) (role : Wrapper.Role) (input : Option Tensor) :
      (resolveOption ft target role input).run=(resolveOption tensor target role input).run := by cases input <;> rfl
  have resolved : (resolveInputs ft q).run=(resolveInputs tensor q).run := by
    unfold resolveInputs
    repeat' first | rfl | apply opt | apply bind_run_congr | intro
  have alloc (role : Wrapper.Role) (condition : Bool) (spec : Wrapper.Allocation) (check : Bool) :
      (allocateOptional ft role condition spec check).run=(allocateOptional tensor role condition spec check).run := by
    cases condition <;> cases check <;> rfl
  have allocated (v : Validated) (input : Wrapper.Inputs) :
      (allocateOutputs ft q v input).run=(allocateOutputs tensor q v input).run := by
    unfold allocateOutputs
    repeat' first | rfl | apply alloc | apply bind_run_congr | intro
  unfold prepare
  repeat' first | rfl | apply validate_run | exact resolved | apply allocated | apply bind_run_congr | intro

theorem framed_arguments_run (ρ : HostState α Host → Address → ℝ)
    (device : DispatchAPI (world (α:=α) (Host:=Host) caller)) (frames : DispatchFrames ρ device)
    (q : Request) (p : Prepared) :
    (arguments (framedDispatch ρ device frames) q p).run=(arguments device q p).run := by
  have opt (role : Wrapper.Role) (value : Option Tensor) :
      (optionalStride (world:=physicalWorld ρ caller) role value).run=
        (optionalStride (world:=world (α:=α) (Host:=Host) caller) role value).run := by cases value <;> rfl
  unfold arguments
  repeat' first | rfl | apply opt | apply bind_run_congr | intro

/-- Full physical prefix frame, including protected addresses with no logical
coordinate. It is derived from the ordinary per-call frames above. -/
theorem prefix_caller_frame
    (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (ρ : HostState α Host → Address → ℝ) (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (laws : Transport store ρ tensor) (tf : TensorFrames ρ tensor)
    (device : DispatchAPI (world (α:=α) (Host:=Host) caller)) (df : DispatchFrames ρ device)
    (q : Request) (initial : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (i : Address) (hi : caller i) :
    interpret ρ (launchState tensor device q initial p args) i=interpret ρ initial i := by
  have hp := (prepare (framedTensor store ρ tensor laws tf) q).frame initial i hi
  change interpret ρ ((prepare (framedTensor store ρ tensor laws tf) q).run initial).state i=_ at hp
  rw [framed_prepare_run] at hp
  have he := df.enter p.inputs.x.device (preparedState tensor q initial) i hi
  have ha := (arguments (framedDispatch ρ device df) q p).frame
    (enteredState device p (preparedState tensor q initial)) i hi
  change interpret ρ ((arguments (framedDispatch ρ device df) q p).run _).state i=_ at ha
  rw [framed_arguments_run] at ha
  have hd := df.admit args (argumentsState device q p (enteredState device p (preparedState tensor q initial))) i hi
  exact hd.trans (ha.trans (he.trans hp))

#axiomsClean prefix_caller_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.RealRevision
