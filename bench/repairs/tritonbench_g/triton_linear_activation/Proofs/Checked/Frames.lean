import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Checked.Host

namespace VeriTile.Bench.Repairs.TritonLinearActivation.Checked
noncomputable section
open Classical
set_option maxHeartbeats 8000000

/-- Compositional state observation, valid for success and ordinary error. -/
theorem bind_observation (observe : DeviceState σ → δ) (result : Result σ β)
    (next : β → DeviceState σ → Result σ γ) (before : DeviceState σ)
    (first : observe result.state = observe before)
    (rest : ∀ value state, observe (next value state).state = observe state) :
    observe (result.bind next).state = observe before := by
  cases result with
  | error why state => exact first
  | ok value state => exact (rest value state).trans first

theorem part_frame (abi : LocalEventABI Word σ a) (raw : ByteABI abi backing modes)
    (fault : Option (Nat×String)) (code : Code Word Unit) (state : DeviceState σ)
    (safe : Code.Safe allowed code) (owner address : Nat)
    (outside : ∀ mode p, allowed mode p → ¬Covers backing mode p owner address) :
    raw.observe (part abi fault code state).state.payload owner address = raw.observe state.payload owner address := by
  cases fault with
  | none => exact native_byte_frame abi raw code state.payload safe owner address outside
  | some pair => exact prefix_byte_frame abi raw code state.payload safe pair.1 owner address outside

theorem launchPart_frame (device : Nat) (abi : LocalEventABI Word σ a) (raw : ByteABI abi backing modes)
    (fault : Option (Nat×String)) (code : Code Word Unit) (state : DeviceState σ)
    (safe : Code.Safe allowed code) (owner address : Nat)
    (outside : ∀ mode p, allowed mode p → ¬Covers backing mode p owner address) :
    raw.observe (launchPart device abi fault code state).state.payload owner address = raw.observe state.payload owner address := by
  unfold launchPart
  split
  · exact part_frame abi raw fault code state safe owner address outside
  · rfl

theorem owner_frame (observe : σ → δ) (owner : Nat) (failures : Failures)
    (body : DeviceState σ → Result σ Unit) (state : DeviceState σ)
    (frame : ∀ s, observe (body s).state.payload = observe s.payload) :
    observe (withOwner owner failures body state).state.payload = observe state.payload := by
  unfold withOwner
  cases enter : failures.enter with
  | some pair => rfl
  | none =>
    have hf := frame {state with currentDevice := owner}
    cases exit : failures.exit with
    | some pair => exact hf
    | none =>
      cases result : body {state with currentDevice := owner} <;> simpa [result,Result.state] using hf

theorem finish_frame (rt : Tail) (h : Allocation r caller (before : Memory α)) (state : DeviceState σ) :
    (finish rt h state).state = state := by
  cases batch : r.batch with
  | matrix m =>
    simp only [finish,batch]
    cases rt.save h.context <;> rfl
  | batched b sequence =>
    simp only [finish,batch]
    split
    · rfl
    · cases rt.save h.context <;> rfl

/-- No success premise: this includes every interrupted cast, autotuner trial,
selected launch, copyback, context entry/exit, reshape and save outcome. -/
theorem run_frame (h : Allocation r caller (before : Memory Word)) (layout : Layout h)
    (abi : LocalEventABI Word σ a) (raw : ByteABI abi layout.backing layout.mode)
    (rt : Tail) (failures : Failures) (trials : List (Fin 98)) (c : Fin 98)
    (state : DeviceState σ) (old owner address : Nat) (live : old ∈ caller)
    (covered : (layout.backing old).covers owner address)
    (outside : ¬SavedCovers r layout.backing owner address) :
    raw.observe (run abi rt h failures trials c state).state.payload owner address =
      raw.observe state.payload owner address := by
  have exterior := allowed_outside layout old owner address live covered outside
  unfold run
  apply bind_observation (fun s => raw.observe s.payload owner address)
  · exact part_frame abi raw _ _ state (safe_prepare h) owner address exterior
  · intro u prepared
    apply bind_observation (fun s => raw.observe s.payload owner address)
    · split
      · exact owner_frame (fun s => raw.observe s owner address) _ _ _ prepared
          (fun s => launchPart_frame r.device abi raw _ _ s (safe_autotuned_kernel h trials c) owner address exterior)
      · rfl
    · intro u launched
      apply bind_observation (fun s => raw.observe s.payload owner address)
      · exact part_frame abi raw _ _ launched (safe_copyback h) owner address exterior
      · intro u copied
        rw [finish_frame]

#axiomsClean run_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonLinearActivation.Checked
