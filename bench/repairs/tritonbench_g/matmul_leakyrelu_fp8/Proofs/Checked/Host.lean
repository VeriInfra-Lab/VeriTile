import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Contract
import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Checked.Bytes
import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Checked.Frames

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Checked
noncomputable section
open Classical
set_option maxHeartbeats 10000000
abbrev DeviceState := TritonLinearActivation.DeviceState
abbrev Result := TritonLinearActivation.Checked.Result
open TritonLinearActivation.Checked (bind_ok bind_observation owner_frame)

structure Failures where
  enter : Option (String×Nat) := none
  exit : Option (String×Nat) := none
  execution : Option (Nat×String) := none

def contextFailures (failures : Failures) : TritonLinearActivation.Checked.Failures :=
  { enter := failures.enter, exit := failures.exit }

def part (abi : LocalABI α σ ops) (fault : Option (Nat×String)) (code : Code α Unit)
    (state : DeviceState σ) : Result σ Unit :=
  match fault with
  | none => .ok () {state with payload := (code.native abi state.payload).2}
  | some (fuel,why) => .error why {state with payload := runPrefix abi fuel code state.payload}

theorem part_success (abi : LocalABI α σ ops) (fault : Option (Nat×String)) (code : Code α Unit)
    (before after : DeviceState σ) (success : part abi fault code before = .ok () after) :
    after.payload = (code.native abi before.payload).2 ∧ after.currentDevice = before.currentDevice := by
  cases fault <;> simp [part] at success
  cases success
  exact ⟨rfl,rfl⟩

def launch (owner : Nat) (abi : LocalABI α σ ops) (fault : Option (Nat×String))
    (code : Code α Unit) (state : DeviceState σ) : Result σ Unit :=
  if state.currentDevice = owner then part abi fault code state else .error "launch owner" state

theorem launch_success (owner : Nat) (abi : LocalABI α σ ops) (fault : Option (Nat×String))
    (code : Code α Unit) (before after : DeviceState σ) (success : launch owner abi fault code before = .ok () after) :
    before.currentDevice = owner ∧ after.payload = (code.native abi before.payload).2 := by
  unfold launch at success
  split at success
  · exact ⟨‹before.currentDevice = owner›,(part_success abi fault code before after success).1⟩
  · cases success

def work (abi : LocalABI α σ ops) (args : Host.Arguments) (output : Tensor) (trials : Nat)
    (failures : Failures) (state : DeviceState σ) : Result σ Tensor :=
  if args.A.rows = 0 ∨ args.B.columns = 0 then .ok output state
  else (if args.A.columns = 0 then part abi failures.execution (computationCode args output trials) state
    else TritonLinearActivation.Checked.withOwner args.A.device (contextFailures failures)
      (launch args.A.device abi failures.execution (computationCode args output trials)) state).bind
        (fun _ after => .ok output after)

theorem work_success (abi : LocalABI α σ ops) (args : Host.Arguments) (output : Tensor) (trials : Nat)
    (failures : Failures) (before after : DeviceState σ) (returned : Tensor)
    (success : work abi args output trials failures before = .ok returned after) :
    returned = output ∧ after.payload = ((computationCode args output trials).native abi before.payload).2 ∧
    after.currentDevice = before.currentDevice := by
  unfold work at success
  split at success
  · have empty := ‹args.A.rows = 0 ∨ args.B.columns = 0›
    cases success
    simp [computationCode,empty,pure,Code.native]
  · obtain ⟨u,done,executed,returnedEq⟩ := (bind_ok _ _ _ _).mp success
    cases u
    cases returnedEq
    refine ⟨rfl,?_⟩
    split at executed
    · exact part_success abi _ _ _ _ executed
    · obtain ⟨inside,launched,values,device⟩ :=
        TritonLinearActivation.Checked.owner_success args.A.device (contextFailures failures) _ before _ executed
      have source := launch_success args.A.device abi _ _ _ _ launched
      exact ⟨values.trans source.2,device⟩

structure Allocator (σ : Type) where
  empty : Nat → Nat → Storage → Nat → DeviceState σ → Result σ Tensor

/-- Allocation metadata and local caller-memory frame, independent of source
values. The observed allocated memory is otherwise arbitrary. -/
structure AllocationLaw (allocate : Allocator σ) (abi : LocalABI α σ ops) (live : σ → Nat → Prop) : Prop where
  layout : ∀ rows columns dtype owner before out after,
    allocate.empty rows columns dtype owner before = .ok out after →
      Host.Allocation live rows columns dtype owner before.payload out after.payload
  device : ∀ rows columns dtype owner before out after,
    allocate.empty rows columns dtype owner before = .ok out after → after.currentDevice = before.currentDevice
  frame : ∀ rows columns dtype owner before out after,
    allocate.empty rows columns dtype owner before = .ok out after → ∀ point, live before.payload point.1 →
      abi.observe after.payload point = abi.observe before.payload point

def call (allocate : Allocator σ) (abi : LocalABI α σ ops) (raw : DynamicArguments)
    (trials : Nat) (failures : Failures) (state : DeviceState σ) : Result σ Tensor :=
  match raw with
  | .tensors args => match checkInputs args with
    | .error why => .error why state
    | .ok () => (allocate.empty args.A.rows args.B.columns .fp16 args.A.device state).bind
      (fun output ready => work abi args output trials failures ready)
  | _ => .error "dynamic argument" state

theorem rejected_state (allocate : Allocator σ) (abi : LocalABI α σ ops) (raw : DynamicArguments)
    (trials : Nat) (failures : Failures) (state : DeviceState σ) (rejected : ¬Admitted raw) :
    (call allocate abi raw trials failures state).state = state := by
  cases raw <;> try rfl
  next args =>
    have no : checkInputs args ≠ .ok () := fun h => rejected ((checks_iff args).mp h)
    simp only [call]
    cases checked : checkInputs args with
    | error why => simp only [checked,TritonLinearActivation.Checked.Result.state]
    | ok u => cases u; exact False.elim (no checked)

theorem allocation_layout {σ : Type} {args : Host.Arguments} {live : σ → Nat → Prop}
    {initial ready : σ} {out : Tensor} (allocation : Host.Allocation live args.A.rows args.B.columns .fp16 args.A.device initial out ready)
    (input : Host.InputContract live args initial) : Layout args.problem ⟨args.A,args.B,out⟩ :=
  ⟨⟨rfl,rfl⟩,⟨input.compatible.symm,rfl⟩,⟨allocation.rowsExact,allocation.columnsExact⟩,
    rfl,input.bType.trans input.aType.symm,allocation.dtype,input.aBound,input.bBound,allocation.bounded,
    ⟨allocation.rowStride,allocation.columnStride⟩,allocation.fresh _ input.aLive,allocation.fresh _ input.bLive⟩

/-- The successful call returns exactly the allocator's Tensor, executes the
complete scalar source stream and restores its actual entry device token. -/
theorem call_success (allocate : Allocator σ) (abi : LocalABI α σ ops) (args : Host.Arguments)
    (law : AllocationLaw allocate abi live) (trials : Nat) (failures : Failures)
    (before after : DeviceState σ) (out : Tensor)
    (success : call allocate abi (.tensors args) trials failures before = .ok out after) :
    MetadataContract args ∧ ∃ ready,
      allocate.empty args.A.rows args.B.columns .fp16 args.A.device before = .ok out ready ∧
      Host.Allocation live args.A.rows args.B.columns .fp16 args.A.device before.payload out ready.payload ∧
      after.payload = ((computationCode args out trials).native abi ready.payload).2 ∧
      after.currentDevice = before.currentDevice := by
  simp only [call] at success
  cases checked : checkInputs args with
  | error why => simp [checked] at success
  | ok u =>
    cases u
    simp only [checked] at success
    obtain ⟨output,ready,allocated,worked⟩ := (bind_ok _ _ _ _).mp success
    obtain ⟨returned,values,device⟩ := work_success abi args output trials failures ready after out worked
    subst output
    exact ⟨(checks_iff args).mp checked,ready,allocated,law.layout _ _ _ _ _ _ _ allocated,
      values,device.trans (law.device _ _ _ _ _ _ _ allocated)⟩

theorem part_frame (abi : LocalABI Word σ ops) (raw : ByteABI abi backing unit)
    (fault : Option (Nat×String)) (code : Code Word Unit) (state : DeviceState σ)
    (safe : Code.Safe allowed code) (owner address : Nat)
    (outside : ∀ point, allowed point → ¬Covers backing point owner address) :
    raw.observe (part abi fault code state).state.payload owner address = raw.observe state.payload owner address := by
  cases fault with
  | none => exact native_frame abi raw code safe state.payload owner address outside
  | some pair => exact prefix_frame abi raw code safe state.payload pair.1 owner address outside

theorem work_frame (abi : LocalABI Word σ ops) (raw : ByteABI abi backing unit)
    (args : Host.Arguments) (output : Tensor) (layout : Layout args.problem ⟨args.A,args.B,output⟩)
    (physical : OutputBacking backing unit live output) (trials : Nat) (failures : Failures)
    (state : DeviceState σ) (old owner address : Nat) (present : live old) (covered : (backing old).covers owner address) :
    raw.observe (work abi args output trials failures state).state.payload owner address = raw.observe state.payload owner address := by
  have outside := output_outside physical old owner address present covered
  have safe := safe_computation (α:=Word) args output layout trials
  unfold work
  split
  · rfl
  · apply bind_observation (fun s => raw.observe s.payload owner address)
    · split
      · exact part_frame abi raw _ _ state safe owner address outside
      · apply owner_frame (fun s => raw.observe s owner address)
        intro s
        unfold launch
        split
        · exact part_frame abi raw _ _ s safe owner address outside
        · rfl
    · intros; rfl

/-- Full fresh-storage and every-outcome caller-byte behavior of the ordinary
allocator. No kernel result or completed source frame is a field. -/
structure AllocationByteLaw (allocate : Allocator σ) (abi : LocalABI Word σ ops)
    (raw : ByteABI abi backing unit) (live : σ → Nat → Prop) : Prop where
  output : ∀ rows columns owner before out after,
    allocate.empty rows columns .fp16 owner before = .ok out after → OutputBacking backing unit (live before.payload) out
  frame : ∀ rows columns owner before old device address, live before.payload old →
    (backing old).covers device address →
    raw.observe (allocate.empty rows columns .fp16 owner before).state.payload device address =
      raw.observe before.payload device address

theorem call_frame (allocate : Allocator σ) (abi : LocalABI Word σ ops) (raw : ByteABI abi backing unit)
    (args : Host.Arguments) (law : AllocationLaw allocate abi live) (bytes : AllocationByteLaw allocate abi raw live)
    (state : DeviceState σ) (aBound : args.A.Bounded) (bBound : args.B.Bounded)
    (aLive : live state.payload args.A.storage) (bLive : live state.payload args.B.storage) (trials : Nat) (failures : Failures) (old owner address : Nat) (present : live state.payload old) (covered : (backing old).covers owner address) :
    raw.observe (call allocate abi (.tensors args) trials failures state).state.payload owner address =
      raw.observe state.payload owner address := by
  simp only [call]
  cases checked : checkInputs args with
  | error why => simp only [checked,TritonLinearActivation.Checked.Result.state]
  | ok u =>
    cases u
    simp only [checked]
    have input := checked_input live args state.payload checked aBound bBound aLive bLive
    have first := bytes.frame args.A.rows args.B.columns args.A.device state old owner address present covered
    cases allocated : allocate.empty args.A.rows args.B.columns .fp16 args.A.device state with
    | error why ready => simpa only [allocated,TritonLinearActivation.Checked.Result.state] using first
    | ok output ready =>
      simp only [allocated,TritonLinearActivation.Checked.Result.state] at first
      have allocation := law.layout _ _ _ _ _ _ _ allocated
      exact (work_frame abi raw args output (allocation_layout allocation input)
        (bytes.output _ _ _ _ _ _ allocated) trials failures ready old owner address present covered).trans first

#axiomsClean call_success
#axiomsClean rejected_state
#axiomsClean work_frame
#axiomsClean call_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Checked
