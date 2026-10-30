import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperRoute
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperNativeAction

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper NativeMachine
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {α Host : Type} {caller : Address → Prop}

/-- Context exit and chunk-view creation do not write tensor storage. These
are ordinary host API effects, with no kernel or output-value premise. -/
structure SuffixMemory (api : DispatchAPI (world (α:=α) (Host:=Host) caller)) : Prop where
  leave : ∀ device s, ((api.leave device).run s).1.memory=s.memory
  split : ∀ tensor s, ((api.split tensor).run s).1.memory=s.memory

def MemoryPreserving (a : Action (world (α:=α) (Host:=Host) caller) β) : Prop :=
  ∀ s, (a.run s).state.memory=s.memory

theorem memory_pure (value : β) : MemoryPreserving (caller:=caller) (Host:=Host) (α:=α) (Action.pure value) :=
  fun _ => rfl

theorem memory_bind (a : Action (world (α:=α) (Host:=Host) caller) β)
    (f : β → Action (world caller) γ) (ha : MemoryPreserving a) (hf : ∀ value, MemoryPreserving (f value)) :
    MemoryPreserving (a.bind f) := by
  intro s
  cases he : (a.run s).result with
  | error e => simpa [Action.bind,he] using ha s
  | ok value => simpa [Action.bind,he] using (hf value (a.run s).state).trans (ha s)

theorem split_memory (api : DispatchAPI (world (α:=α) (Host:=Host) caller))
    (hm : SuffixMemory api) (p : Prepared) : MemoryPreserving (splitMasks api p) := by
  unfold splitMasks
  cases hmask : p.outputs.mask <;> cases hx : p.inputs.x1
  all_goals first | exact memory_pure _ | skip
  exact memory_bind _ _ (hm.split _) (fun _ => memory_pure _)

theorem finish_memory (api : DispatchAPI (world (α:=α) (Host:=Host) caller))
    (hm : SuffixMemory api) (p : Prepared) : MemoryPreserving (finish api p) :=
  memory_bind _ _ (split_memory api hm p) (fun _ => memory_bind _ _ (fun _ => rfl) (fun _ => memory_pure _))

/-- Successful cleanup and view splitting retain the actual kernel's entire
physical final memory, including cells outside returned tensor coordinates. -/
theorem route_final_memory (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (device : DispatchAPI (world caller)) (hm : SuffixMemory device)
    (kernel : LaunchArguments → Action (world caller) Unit) (q : Request) (s : HostState α Host)
    (result : Returned) (p : Prepared) (args : LaunchArguments)
    (route : SuccessRoute tensor device kernel q s result p args) :
    ((surround tensor device kernel q).run s).state.memory=
      ((kernel args).run (admittedState device args (argumentsState device q p
        (enteredState device p (preparedState tensor q s))))).state.memory := by
  rw [route.finalState]
  rw [finish_memory device hm p]
  exact hm.leave _ _

theorem action_success_history [Inhabited α] (c : Config α block) (hf : CallerFresh c caller)
    (benchmarks : List (DispatchEvent block)) (selected : Schedule block) (s : HostState α Host)
    (h : ((dispatchAction c caller hf benchmarks selected).run s).result=.ok ()) :
    (history c benchmarks s.memory).error=none := by
  cases he : (history c benchmarks s.memory).error with
  | none => rfl
  | some code =>
    rw [action_failure c caller hf benchmarks selected s code he] at h
    cases h

#axiomsClean memory_bind
#axiomsClean split_memory
#axiomsClean finish_memory
#axiomsClean route_final_memory
#axiomsClean action_success_history
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
