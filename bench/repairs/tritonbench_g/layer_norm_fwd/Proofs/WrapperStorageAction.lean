import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperStorageReady
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.StorageHistory

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperStorage
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper WrapperNative
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000
variable {α Host : Type} [Inhabited α] {caller : Address → Prop}

def compiledEvent {compiler : StorageCompiler.CompilerAPI Host} {args : LaunchArguments} {host : Host} :
    StorageCompiler.ProducedEvent compiler args host →
      StorageMachine.DispatchEvent (StorageCompiler.Profile.ofRequest args.request)
        (args.prepared.flags args.request) args.prepared.validated.N args.prepared.block
  | .launch produced => .launch produced.artifact
  | .setupFailure code => .setupFailure code

def compiledBenchmarks {compiler : StorageCompiler.CompilerAPI Host} {args : LaunchArguments} {host : Host}
    (produced : StorageCompiler.ProducedDispatch compiler args host) :=
  produced.benchmarks.map compiledEvent

def dispatchAction (c : StorageMachine.Config α block) (caller : Address → Prop)
    (hf : ∀ i, caller i → ∀ k,
      AliasMemory.Writes c.flags preparedRegions c.strides c.M c.N k →
      (PhysicalDomain.layout c.allocations).address k≠i.val)
    (benchmarks : List (StorageMachine.DispatchEvent c.profile c.flags c.N block))
    (selected : StorageCompiler.Artifact c.profile c.flags c.N block) : Action (world (α:=α) (Host:=Host) caller) Unit where
  run s :=
    let result := StorageMachine.dispatch c benchmarks selected s.memory
    ⟨{s with memory:=result.memory},
      match result.error with | none => .ok () | some code => .error (.api .execute code),[]⟩
  frame s i hi := StorageMachine.dispatch_frame c benchmarks selected s.memory i (hf i hi)

def action (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (primitives : Compiler.FP32Primitives α) (casts : StorageCompiler.StoreCasts α)
    (args : LaunchArguments) (current : HostState α Host) (ready : PreparedReady store args current caller)
    {compiler : StorageCompiler.CompilerAPI Host} (produced : StorageCompiler.ProducedDispatch compiler args current.host) :
    Action (world (α:=α) (Host:=Host) caller) Unit :=
  dispatchAction (config store primitives casts args current ready) caller ready.outside
    (compiledBenchmarks produced) produced.selected.artifact

/-- Concrete partial kernel family. Its readiness guard is derived for every
admitted default-input request, including both explicit output dtype routes.
Only actual compile/setup failures remain ordinary failing outcomes. -/
def kernel (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (primitives : Compiler.FP32Primitives α) (casts : StorageCompiler.StoreCasts α)
    (selection : StorageCompiler.ProducedSelection Host) (args : LaunchArguments) :
    Action (world (α:=α) (Host:=Host) caller) Unit where
  run current := match selection.select args current.host with
    | .error e => ⟨current,.error e,[]⟩
    | .ok produced => if h : PreparedReady store args current caller then
        (action store primitives casts args current h produced).run current
      else ⟨current,.error (.api .admission 0),[]⟩
  frame current i hi := by
    cases hs : selection.select args current.host with
    | error e => simp [hs]
    | ok produced =>
      simp only [hs]
      split
      · exact (action store primitives casts args current _ produced).frame current i hi
      · rfl

theorem kernel_configured (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (primitives : Compiler.FP32Primitives α) (casts : StorageCompiler.StoreCasts α)
    (selection : StorageCompiler.ProducedSelection Host) (args : LaunchArguments) (current : HostState α Host)
    (ready : PreparedReady store args current caller)
    (produced : StorageCompiler.ProducedDispatch selection.compiler args current.host)
    (hselect : selection.select args current.host=.ok produced) :
    ((kernel (caller:=caller) store primitives casts selection args).run current)=
      (action store primitives casts args current ready produced).run current := by
  simp [kernel,hselect,ready]

theorem kernel_success (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (primitives : Compiler.FP32Primitives α) (casts : StorageCompiler.StoreCasts α)
    (selection : StorageCompiler.ProducedSelection Host) (args : LaunchArguments) (current : HostState α Host)
    (ready : PreparedReady store args current caller)
    (h : ((kernel (caller:=caller) store primitives casts selection args).run current).result=.ok ()) :
    ∃ produced, selection.select args current.host=.ok produced ∧
      ((action store primitives casts args current ready produced).run current).result=.ok () ∧
      ((kernel (caller:=caller) store primitives casts selection args).run current)=
        (action store primitives casts args current ready produced).run current := by
  cases hs : selection.select args current.host with
  | error e => simp [kernel,hs] at h
  | ok produced =>
    have he := kernel_configured store primitives casts selection args current ready produced hs
    exact ⟨produced,rfl,by simpa only [he] using h,he⟩

theorem action_success_history (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (primitives : Compiler.FP32Primitives α) (casts : StorageCompiler.StoreCasts α)
    (args : LaunchArguments) (current : HostState α Host) (ready : PreparedReady store args current caller)
    {compiler : StorageCompiler.CompilerAPI Host} (produced : StorageCompiler.ProducedDispatch compiler args current.host)
    (h : ((action store primitives casts args current ready produced).run current).result=.ok ()) :
    (StorageMachine.history (config store primitives casts args current ready)
      (compiledBenchmarks produced) current.memory).error=none := by
  cases he : (StorageMachine.history (config store primitives casts args current ready)
      (compiledBenchmarks produced) current.memory).error with
  | none => rfl
  | some code => simp [action,dispatchAction,StorageMachine.dispatch,he] at h

#axiomsClean kernel_configured
#axiomsClean kernel_success
#axiomsClean action_success_history
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperStorage
