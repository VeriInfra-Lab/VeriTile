import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeMachineDispatch
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperProtocol

/-! A concrete host action for the admitted native execution. Host allocation,
device and tensor-descriptor metadata is carried separately from physical
cells. The action computes benchmark and selected-launch outputs itself. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open NativeMachine Wrapper
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000

abbrev Address := Fin (2^64)
abbrev Memory (α : Type) := Address → Cell α

structure HostState (α Host : Type) where
  memory : Memory α
  host : Host

def world {α Host : Type} (caller : Address → Prop) : World (HostState α Host) Address (Cell α) where
  observe s i := s.memory i
  caller := caller

/-- Ordinary fresh-output allocation condition relative to the cells that
existed before the wrapper call. This is an address condition, not an execution
frame or successful-output premise. -/
def CallerFresh {α : Type} (c : Config α block) (caller : Address → Prop) : Prop :=
  ∀ i, caller i → ∀ row : Fin c.M, Outside c row i

def dispatchAction {α Host : Type} [Inhabited α] (c : Config α block)
    (caller : Address → Prop) (hf : CallerFresh c caller)
    (benchmarks : List (DispatchEvent block)) (selected : Schedule block) :
    Action (world (α:=α) (Host:=Host) caller) Unit where
  run s :=
    let result := dispatch c benchmarks selected s.memory
    ⟨{s with memory:=result.memory},
      match result.error with | none => .ok () | some code => .error (.api .execute code),[]⟩
  frame s i hi := dispatch_frame c benchmarks selected s.memory i (hf i hi)

theorem action_state {α Host : Type} [Inhabited α] (c : Config α block)
    (caller : Address → Prop) (hf : CallerFresh c caller)
    (benchmarks : List (DispatchEvent block)) (selected : Schedule block) (s : HostState α Host) :
    ((dispatchAction c caller hf benchmarks selected).run s).state=
      {s with memory:=(dispatch c benchmarks selected s.memory).memory} := rfl

/-- A setup error returns the real benchmark-prefix memory, even if earlier
benchmarks have already changed the fresh output allocations. -/
theorem action_failure {α Host : Type} [Inhabited α] (c : Config α block)
    (caller : Address → Prop) (hf : CallerFresh c caller)
    (benchmarks : List (DispatchEvent block)) (selected : Schedule block) (s : HostState α Host)
    (code : Nat) (h : (history c benchmarks s.memory).error=some code) :
    (dispatchAction c caller hf benchmarks selected).run s=
      ⟨{s with memory:=(history c benchmarks s.memory).memory},.error (.api .execute code),[]⟩ := by
  simp [dispatchAction,dispatch_failure c benchmarks selected s.memory code h,h]

/-- Successful admitted execution retains host metadata and returns the
checked complete machine output from the original pre-benchmark input state. -/
theorem action_return {α Host : Type} [Inhabited α] (c : Config α block)
    (caller : Address → Prop) (hf : CallerFresh c caller)
    (hsep : Domain.ReadWriteSeparated c.flags c.regions)
    (hgeo : NativeGeometry.Geometry c.flags c.regions c.strides c.M c.N)
    (benchmarks : List (DispatchEvent block)) (selected : Schedule block) (s : HostState α Host)
    (template : State α) (h : (history c benchmarks s.memory).error=none) :
    ((dispatchAction c caller hf benchmarks selected).run s).result=.ok () ∧
    ((dispatchAction c caller hf benchmarks selected).run s).state.host=s.host ∧
    TypedSource.observe c.flags c.regions c.strides c.M c.N
      (view c ((dispatchAction c caller hf benchmarks selected).run s).state.memory template)=
      NativeSchedule.oracle c.ops selected.reductions selected.plan selected.plan1 c.flags c.regions c.strides
        c.M c.N c.eps c.p (view c s.memory template) := by
  refine ⟨?_,rfl,?_⟩
  · simp [dispatchAction,dispatch_success c benchmarks selected s.memory h]
  · exact dispatch_return c hsep hgeo benchmarks selected s.memory template h

theorem action_caller_frame {α Host : Type} [Inhabited α] (c : Config α block)
    (caller : Address → Prop) (hf : CallerFresh c caller)
    (benchmarks : List (DispatchEvent block)) (selected : Schedule block) (s : HostState α Host)
    (i : Address) (hi : caller i) :
    ((dispatchAction c caller hf benchmarks selected).run s).state.memory i=s.memory i :=
  (dispatchAction c caller hf benchmarks selected).frame s i hi

#axiomsClean action_state
#axiomsClean action_failure
#axiomsClean action_return
#axiomsClean action_caller_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
