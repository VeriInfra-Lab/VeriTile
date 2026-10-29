import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Checked.Frames
import bench.repairs.tritonbench_g.triton_linear_activation.Kernels

namespace VeriTile.Bench.Repairs.TritonLinearActivation.Checked
noncomputable section
open Classical
set_option maxHeartbeats 8000000

/-- Validation runs before the real allocator; ordinary allocation failure is
propagated with its actual state. Successful allocation feeds the actual
cast/kernel/copyback/context/return program. -/
def call (ts : Fin 4 → Observed) (activationIsValid : Bool)
    (allocate : DeviceState σ → Result σ Unit) (abi : LocalEventABI α σ a)
    (rt : Tail) (h : Allocation r caller (before : Memory α)) (failures : Failures)
    (trials : List (Fin 98)) (c : Fin 98) (state : DeviceState σ) : Result σ Tensor :=
  match check r ts activationIsValid with
  | .error why => .error why state
  | .ok () => (allocate state).bind fun _ ready => run abi rt h failures trials c ready

/-- Ordinary allocator interface: fresh contents may be arbitrary. Its only
value law records its observed memory, and its device law is host metadata. -/
structure AllocationLaw (abi : LocalEventABI α σ a) (h : Allocation r caller (before : Memory α))
    (allocate : DeviceState σ → Result σ Unit) : Prop where
  observed : ∀ state ready, abi.observe state.payload = before →
    allocate state = .ok () ready → abi.observe ready.payload = h.initial
  device : ∀ state ready, allocate state = .ok () ready → ready.currentDevice = state.currentDevice

/-- Actual raw backing and pointer observations for caller objects. Capacities,
address range and dtype are ordinary Tensor-storage facts, not numeric ports. -/
structure PhysicalBinding (h : Allocation r caller (before : Memory Word)) (layout : Layout h)
    (ts : Fin 4 → Observed) : Prop where
  identity : Binding r ts
  backing : ∀ i, Used r i → (layout.backing (r.storage i)).owner = (ts i).metadata.device.index ∧
    (layout.backing (r.storage i)).baseByte = (ts i).base ∧
    (layout.backing (r.storage i)).bytes = (ts i).bytes
  pointer : ∀ i, Used r i → (ts i).metadata.numel ≠ 0 → (ts i).metadata.dataPtr.toNat = (ts i).base
  capacity : ∀ i, Used r i → 4*(ts i).metadata.numel ≤ (ts i).bytes
  fp32 : ∀ i, Used r i → layout.mode (r.storage i) = .fp32
  nativeExtent : ∀ storage, (layout.backing storage).baseByte+(layout.backing storage).bytes ≤ 2^64

/-- Allocation can change arbitrary fresh bytes on either outcome. No caller
byte may change, including saved bytes and untyped partial backing tails. -/
structure AllocationByteLaw (h : Allocation r caller (before : Memory Word)) (layout : Layout h)
    (abi : LocalEventABI Word σ a) (raw : ByteABI abi layout.backing layout.mode)
    (allocate : DeviceState σ → Result σ Unit) : Prop where
  frame : ∀ state old owner address, old ∈ caller →
    (layout.backing old).covers owner address →
    raw.observe (allocate state).state.payload owner address = raw.observe state.payload owner address

theorem rejected_state (ts : Fin 4 → Observed) (flag : Bool)
    (allocate : DeviceState σ → Result σ Unit) (abi : LocalEventABI α σ a)
    (rt : Tail) (h : Allocation r caller (before : Memory α)) (failures : Failures)
    (trials : List (Fin 98)) (c : Fin 98) (state : DeviceState σ)
    (rejected : ¬Domain r ts flag) : (call ts flag allocate abi rt h failures trials c state).state = state := by
  have no : check r ts flag ≠ .ok () := fun checked => rejected ((python_checks_iff r ts flag).mp checked)
  unfold call
  cases checked : check r ts flag with
  | error why => rfl
  | ok u => cases u; exact False.elim (no checked)

theorem call_success (ts : Fin 4 → Observed) (flag : Bool) (binding : Binding r ts)
    (allocate : DeviceState σ → Result σ Unit) (abi : LocalEventABI α σ a)
    (rt : Tail) (h : Allocation r caller (before : Memory α))
    (allocLaw : AllocationLaw abi h allocate) (tailLaw : TailContract rt h)
    (failures : Failures) (trials : List (Fin 98)) (c : Fin 98)
    (state after : DeviceState σ) (out : Tensor) (entry : abi.observe state.payload = before)
    (success : call ts flag allocate abi rt h failures trials c state = .ok out after) :
    Domain r ts flag ∧ out = h.returned ∧
    abi.observe after.payload = h.finalMemory a c ∧ after.currentDevice = state.currentDevice ∧
    rt.save h.context = .ok () := by
  unfold call at success
  cases checked : check r ts flag with
  | error why => simp [checked] at success
  | ok u =>
    cases u
    simp only [checked] at success
    have admit := (python_checks_iff r ts flag).mp checked
    obtain ⟨u,ready,allocated,ran⟩ := (bind_ok _ _ _ _).mp success
    cases u
    obtain ⟨returned,payload,device,saved⟩ := run_success abi rt h tailLaw failures trials c ready after out ran
    refine ⟨admit,returned,?_,device.trans (allocLaw.device state ready allocated),saved⟩
    rw [payload]
    exact h.native_autotuned a trials c (domain_valid r ts binding admit) abi ready.payload
      (allocLaw.observed state ready entry allocated)

/-- Public pre-call byte frame for every return path. The frame is derived
from one-event spans and allocator freshness, never assumed for a whole call. -/
theorem call_frame (ts : Fin 4 → Observed) (flag : Bool)
    (allocate : DeviceState σ → Result σ Unit) (h : Allocation r caller (before : Memory Word))
    (layout : Layout h) (abi : LocalEventABI Word σ a) (raw : ByteABI abi layout.backing layout.mode)
    (allocLaw : AllocationByteLaw h layout abi raw allocate)
    (rt : Tail) (failures : Failures) (trials : List (Fin 98)) (c : Fin 98)
    (state : DeviceState σ) (old owner address : Nat) (live : old ∈ caller)
    (covered : (layout.backing old).covers owner address)
    (outside : ¬SavedCovers r layout.backing owner address) :
    raw.observe (call ts flag allocate abi rt h failures trials c state).state.payload owner address =
      raw.observe state.payload owner address := by
  unfold call
  cases check r ts flag with
  | error why => rfl
  | ok u =>
    cases u
    apply bind_observation (fun s => raw.observe s.payload owner address)
    · exact allocLaw.frame state old owner address live covered
    · intro u ready
      exact run_frame h layout abi raw rt failures trials c ready old owner address live covered outside

#axiomsClean rejected_state
#axiomsClean call_success
#axiomsClean call_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonLinearActivation.Checked
