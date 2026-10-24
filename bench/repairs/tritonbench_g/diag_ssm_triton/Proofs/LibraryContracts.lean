import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceActions
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.NativeBinding

/-! Ordinary local tensor-library contracts. They say nothing about SSM output
values, successful kernel execution or kernel frames. In particular an allocator
provides fresh actual storage, not initialized values for empty_like. A sum
provides its actual FP addition tree, not an equality between rounding orders. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.LibraryContracts
open Kernels ByteMemory Execution Protocol Tensors Host
set_option autoImplicit false

abbrev Succeeded {α : Type} (program : Program State α) (before : State) (value : α) (after : State) :=
  ∃ trace, program before = .success value after trace

@[simp] theorem succeeded_pure {α : Type} (value result : α) (before after : State) :
    Succeeded (Protocol.pure value) before result after ↔ result = value ∧ after = before := by
  simp [Succeeded, Protocol.pure]
  exact ⟨fun h => ⟨h.1.symm, h.2.symm⟩, fun h => ⟨h.1.symm, h.2.symm⟩⟩

@[simp] theorem succeeded_call {α : Type} (label : Call) (program : Program State α)
    (before after : State) (value : α) :
    Succeeded (call label program) before value after ↔ Succeeded program before value after := by
  cases h : program before <;> simp [Succeeded, call, h, Result.prepend]

@[simp] theorem succeeded_bind {α β : Type} (first : Program State α) (next : α → Program State β)
    (before after : State) (value : β) :
    Succeeded (Protocol.bind first next) before value after ↔
      ∃ intermediate mid, Succeeded first before intermediate mid ∧ Succeeded (next intermediate) mid value after := by
  cases h : first before with
  | failure error failed trace => simp [Succeeded, Protocol.bind, h]
  | success v mid trace =>
    cases hn : next v mid <;> simp [Succeeded, Protocol.bind, h, hn, Result.prepend]

abbrev Resident (state : State) (tensor : Tensor) := tensor.view.backing ∈ state.live

structure Extension (before after : State) : Prop where
  live : ∀ backing ∈ before.live, backing ∈ after.live
  frame : ∀ backing ∈ before.live, Frame before.heap after.heap backing

theorem Extension.refl (state : State) : Extension state state := ⟨fun _ h => h, fun _ _ => Frame.refl _ _⟩

theorem Extension.trans {a b c : State} (ab : Extension a b) (bc : Extension b c) : Extension a c :=
  ⟨fun v hv => bc.live v (ab.live v hv), fun v hv => (ab.frame v hv).trans (bc.frame v (ab.live v hv))⟩

theorem Extension.resident {before after : State} {tensor : Tensor}
    (extension : Extension before after) (resident : Resident before tensor) : Resident after tensor :=
  extension.live _ resident

/-- Every API outcome preserves prior live tensor bytes, even if a new owned
allocation was partially written before an exception. Successful calls also
satisfy their local reply contract. -/
structure Ordinary {α : Type} (program : Program State α)
    (reply : State → α → State → Prop) : Prop where
  effect : ∀ before, Extension before (program before).state
  trace : ∀ before, (program before).trace = []
  success : ∀ before value after, Succeeded program before value after → reply before value after

theorem Ordinary.extension {α : Type} {program : Program State α}
    {reply : State → α → State → Prop} (contract : Ordinary program reply)
    {before after : State} {value : α} (run : Succeeded program before value after) : Extension before after := by
  rcases run with ⟨trace, run⟩
  simpa [run, Result.state] using contract.effect before

structure Allocation (source result : Tensor) (before after : State) : Prop where
  same : SameShape source result
  dense : Dense result
  resolved : Resolved result
  resident : Resident after result
  fresh : ∀ backing ∈ before.live, result.view.backing.disjoint backing
  metadata : NativeBinding.Metadata result
  aligned : 0 < result.count → result.view.pointer % 4 = 0
  device : result.device = source.device

structure SavedReply (s y transition : Tensor) (before after : State) : Prop where
  saved : after.saved = some ⟨s, y, transition, before.heap⟩

structure Restored (before : State) (result : Tensor × Tensor × Tensor) (after : State) where
  saved : Saved
  present : before.saved = some saved
  initial : SameSavedStorage saved.initial result.1
  output : SameSavedStorage saved.output result.2.1
  transition : SameSavedStorage saved.transition result.2.2
  initialDevice : result.1.device = saved.initial.device
  outputDevice : result.2.1.device = saved.output.device
  transitionDevice : result.2.2.device = saved.transition.device
  initialDense : Dense saved.initial → Dense result.1
  outputDense : Dense saved.output → Dense result.2.1
  transitionDense : Dense saved.transition → Dense result.2.2
  initialResident : Resident after result.1
  outputResident : Resident after result.2.1
  transitionResident : Resident after result.2.2
  initialVersion : before.version saved.initial.storage = saved.initial.version
  outputVersion : before.version saved.output.storage = saved.output.version
  transitionVersion : before.version saved.transition.storage = saved.transition.version

structure SumReply (ops : Operations Word) (codec : Codec Word) (input result : Tensor)
    (before after : State) where
  batch : Nat
  dim : Nat
  inputShape : input.shape = [batch, dim]
  device : result.device = input.device
  route : result.route = input.route
  shape : result.shape = [dim]
  dense : Dense result
  resolved : Resolved result
  resident : Resident after result
  fresh : ∀ backing ∈ before.live, result.view.backing.disjoint backing
  plan : Reduction.Plan batch dim input.route.components
  allocated : Heap
  allocationFrame : ∀ backing ∈ before.live, Frame before.heap allocated backing
  actual : after.heap = Reduction.execute codec input.route ops batch dim plan input.view result.view allocated

structure Contracts (codec : Codec Word) (ops : Operations Word) (api : Services) : Prop where
  contiguous : ∀ tensor, Ordinary (api.contiguous tensor) (fun before result after =>
    ContiguousReply codec ops.flipSign tensor result before.heap after.heap ∧ Resident after result)
  zeros : ∀ tensor, Ordinary (api.zerosLike tensor) (fun before result after => Allocation tensor result before after)
  empty : ∀ tensor, Ordinary (api.emptyLike tensor) (fun before result after => Allocation tensor result before after)
  realView : ∀ tensor, Ordinary (api.realView tensor) (fun _ result _ => result = tensor.view)
  prepare : ∀ backward route device, Ordinary (api.prepare backward route device) (fun _ _ _ => True)
  save : ∀ s y transition, Ordinary (api.save s y transition) (fun before _ after => SavedReply s y transition before after)
  restore : Ordinary api.restore (fun before result after => Nonempty (Restored before result after))
  savedVersionFailure : ∀ before saved, before.saved = some saved →
    (before.version saved.initial.storage ≠ saved.initial.version ∨
     before.version saved.output.storage ≠ saved.output.version ∨
     before.version saved.transition.storage ≠ saved.transition.version) →
    api.restore before = .failure .savedVersion before []
  sum : ∀ tensor, Ordinary (api.sumDim0 tensor) (fun before result after => Nonempty (SumReply ops codec tensor result before after))

/-- Admission returns the original descriptor and heap. Residency is an
ordinary caller ownership fact, independent of the checked input domain. -/
theorem normalize_identity (api : Services) (role : Role) (tensor result : Tensor)
    (before after : State)
    (run : Succeeded (Host.normalize api role tensor) before result after) :
    result = tensor ∧ after = before ∧ Resolved tensor := by
  simp only [Host.normalize, succeeded_call] at run
  split at run
  · simp [Succeeded, raise] at run
  · rename_i flags
    have ids := (succeeded_pure _ _ _ _).mp run
    exact ⟨ids.1, ids.2, by simpa [Resolved] using flags⟩

theorem normalize_success (codec : Codec Word) (ops : Operations Word) (api : Services)
    (_contracts : Contracts codec ops api) (role : Role) (tensor result : Tensor)
    (before after : State) (dense : Dense tensor) (live : Resident before tensor)
    (run : Succeeded (Host.normalize api role tensor) before result after) :
    Extension before after ∧ SameLogical codec ops.flipSign tensor result before.heap after.heap ∧
      Dense result ∧ Resolved result ∧ Resident after result := by
  obtain ⟨rfl, rfl, flags⟩ := normalize_identity api role tensor result before after run
  exact ⟨Extension.refl _, ⟨⟨rfl, rfl⟩, fun _ _ _ _ => rfl⟩, dense, flags, live⟩

theorem normalize_effect (codec : Codec Word) (ops : Operations Word) (api : Services)
    (_contracts : Contracts codec ops api) (role : Role) (tensor : Tensor) (before : State) :
    Extension before (Host.normalize api role tensor before).state := by
  unfold Host.normalize
  split <;> exact Extension.refl before

theorem invalid_saved_version (codec : Codec Word) (ops : Operations Word) (api : Services)
    (contracts : Contracts codec ops api) (kernels : KernelActions) (gy : Tensor)
    (before : State) (saved : Saved) (present : before.saved = some saved)
    (changed : before.version saved.initial.storage ≠ saved.initial.version ∨
     before.version saved.output.storage ≠ saved.output.version ∨
     before.version saved.transition.storage ≠ saved.transition.version) :
    Host.backward api kernels gy before = .failure .savedVersion before [.restore] := by
  simp [Host.backward, call, contracts.savedVersionFailure before saved present changed,
    Protocol.bind, Result.prepend]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.LibraryContracts
