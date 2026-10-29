import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.Index
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.Numeric

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization
noncomputable section
set_option maxHeartbeats 5000000

/-- Explicit ordinary source/frontend/compiler binding. The lowered program
is fixed from actual source coordinates, local operations and commit order.
Its fields contain no numeric output or memory/frame equality. -/
structure BoundCall (I F H : Type) (s : Shape) (b : Buffers) where
  invocation : Invocation s
  parsedProduction : String
  sourceIdentity : parsedProduction=Source.productionAST
  admitted : Source.Admitted invocation.config
  indices : IndexABI s invocation.config b
  dispatch : invocation.schedule.events.Perm (allCommits s invocation.config)
  lowered : Code I F H Unit
  sourceLowering : lowered=gridCode s invocation.config b invocation.selected invocation.schedule.events
structure BoundQuantCall (I F H : Type) (G K : Nat) (q : QuantBuffers) where
  invocation : QuantInvocation G K
  parsedProduction : String
  sourceIdentity : parsedProduction=Source.productionAST
  configIndex : Fin 4
  block : Nat
  indices : QuantIndexABI G K block q
  rowStructure : q.transposeOutput=Bool.false ∧ q.scalePrecision=.half ∧ q.initialMaximumZero=Bool.true
  lowered : Code I F H Unit
  sourceLowering : lowered=quantGridCode q invocation.plan invocation.groups

structure ZeroHookABI (abi : LocalEventABI I F H σ a) where
  zero : Shape → Buffers → σ → σ
  law : ∀ s b state,abi.observe (zero s b state)=reset a s b (abi.observe state)
def compiledGrid (abi : LocalEventABI I F H σ a) (call : BoundCall I F H s b) (state : σ) : σ :=
  (call.lowered.native abi state).2
theorem compiledGrid_refines (abi : LocalEventABI I F H σ a) (call : BoundCall I F H s b) (state : σ) :
    abi.observe (compiledGrid abi call state)=
      gridRun a s call.invocation.config b call.invocation.selected call.invocation.schedule.events (abi.observe state) := by
  unfold compiledGrid
  rw [call.sourceLowering]
  have h := congrArg Prod.snd ((gridCode s call.invocation.config b call.invocation.selected call.invocation.schedule.events).native_refines abi state)
  simpa only [Prod.snd,eval_gridCode] using h
def compiledTrial (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (call : BoundCall I F H s b) (state : σ) := compiledGrid abi call (hook.zero s b state)
theorem compiledTrial_refines (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (call : BoundCall I F H s b) (state : σ) :
    abi.observe (compiledTrial abi hook call state)=invoke a s b call.invocation (abi.observe state) := by
  rw [compiledTrial,compiledGrid_refines,hook.law]
  rfl
def compiledTrials (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi) :
    List (BoundCall I F H s b) → σ → σ
  | [],state => state
  | c::cs,state => compiledTrials abi hook cs (compiledTrial abi hook c state)
theorem compiledTrials_refines (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (calls : List (BoundCall I F H s b)) (state : σ) :
    abi.observe (compiledTrials abi hook calls state)=trials a s b (calls.map BoundCall.invocation) (abi.observe state) := by
  induction calls generalizing state with
  | nil => rfl
  | cons c cs ih => rw [compiledTrials,ih,compiledTrial_refines]; rfl
def compiledRuntime (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (calls : List (BoundCall I F H s b)) (chosen : BoundCall I F H s b) (state : σ) : σ :=
  let initial := hook.zero s b state
  let done := compiledTrials abi hook calls initial
  let ready := if calls=[] then done else hook.zero s b done
  compiledGrid abi chosen ready
theorem compiledRuntime_refines (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (calls : List (BoundCall I F H s b)) (chosen : BoundCall I F H s b) (state : σ) :
    abi.observe (compiledRuntime abi hook calls chosen state)=
      runtime a s b (calls.map BoundCall.invocation) chosen.invocation (abi.observe state) := by
  unfold compiledRuntime runtime
  rw [compiledGrid_refines]
  cases calls <;> simp only [List.map_nil,List.map_cons,List.cons_ne_self,ite_true,reduceCtorEq,ite_false,
    hook.law,compiledTrials_refines]

structure NativeReturned (σ : Type) where
  tensor : Tensor
  state : σ
  currentDevice : Nat
def compiledWrapper (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (owner callerDevice : Nat) (calls : List (BoundCall I F H s b))
    (chosen : BoundCall I F H s b) (state : σ) : NativeReturned σ :=
  if s.M=0 ∨ s.N=0 then ⟨outputTensor s b owner,hook.zero s b state,callerDevice⟩
  else
    let context := DequantizeMatmul.deviceCall owner
      (fun entered => ⟨entered.currentDevice,compiledRuntime abi hook calls chosen entered.payload⟩)
      ⟨callerDevice,state⟩
    ⟨outputTensor s b owner,context.payload,context.currentDevice⟩
theorem compiledWrapper_refines (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (owner callerDevice : Nat) (calls : List (BoundCall I F H s b))
    (chosen : BoundCall I F H s b) (state : σ) :
    let actual := compiledWrapper abi hook owner callerDevice calls chosen state
    let reference := wrapper a s b owner callerDevice (calls.map BoundCall.invocation) chosen.invocation (abi.observe state)
    (actual.tensor,abi.observe actual.state,actual.currentDevice)=(reference.tensor,reference.memory,reference.currentDevice) := by
  unfold compiledWrapper wrapper
  split_ifs
  · simp only [hook.law]
  · simp only [DequantizeMatmul.deviceCall,compiledRuntime_refines]

def compiledQuantRun (abi : LocalEventABI I F H σ a) (call : BoundQuantCall I F H G K q) (state : σ) : σ :=
  (call.lowered.native abi state).2
theorem compiledQuantRun_refines (abi : LocalEventABI I F H σ a) (call : BoundQuantCall I F H G K q) (state : σ) :
    abi.observe (compiledQuantRun abi call state)=quantRun a q call.invocation (abi.observe state) := by
  unfold compiledQuantRun
  rw [call.sourceLowering]
  exact nativeQuantRun_refines abi q call.invocation state
def compiledQuantTrials (abi : LocalEventABI I F H σ a) : List (BoundQuantCall I F H G K q) → σ → σ
  | [],state => state
  | c::cs,state => compiledQuantTrials abi cs (compiledQuantRun abi c state)
theorem compiledQuantTrials_refines (abi : LocalEventABI I F H σ a)
    (calls : List (BoundQuantCall I F H G K q)) (state : σ) :
    abi.observe (compiledQuantTrials abi calls state)=quantTrials a q (calls.map BoundQuantCall.invocation) (abi.observe state) := by
  induction calls generalizing state with
  | nil => rfl
  | cons c cs ih => rw [compiledQuantTrials,ih,compiledQuantRun_refines]; rfl
def compiledQuantWrapper (abi : LocalEventABI I F H σ a)
    (calls : List (BoundQuantCall I F H G K q)) (chosen : BoundQuantCall I F H G K q) (state : σ) :=
  compiledQuantRun abi chosen (compiledQuantTrials abi calls state)
theorem compiledQuantWrapper_refines (abi : LocalEventABI I F H σ a)
    (calls : List (BoundQuantCall I F H G K q)) (chosen : BoundQuantCall I F H G K q) (state : σ) :
    abi.observe (compiledQuantWrapper abi calls chosen state)=
      quantWrapper a q (calls.map BoundQuantCall.invocation) chosen.invocation (abi.observe state) := by
  rw [compiledQuantWrapper,compiledQuantRun_refines,compiledQuantTrials_refines]
  rfl
def compiledComposed (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (link : CompositionLink q b)
    (qtrials : List (BoundQuantCall I F H s.M s.K q)) (qchosen : BoundQuantCall I F H s.M s.K q)
    (owner callerDevice : Nat) (calls : List (BoundCall I F H s b))
    (chosen : BoundCall I F H s b) (state : σ) : NativeReturned σ :=
  compiledWrapper abi hook owner callerDevice calls chosen (compiledQuantWrapper abi qtrials qchosen state)
theorem compiledComposed_refines (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (link : CompositionLink q b)
    (qtrials : List (BoundQuantCall I F H s.M s.K q)) (qchosen : BoundQuantCall I F H s.M s.K q)
    (owner callerDevice : Nat) (calls : List (BoundCall I F H s b))
    (chosen : BoundCall I F H s b) (state : σ) :
    let actual := compiledComposed abi hook link qtrials qchosen owner callerDevice calls chosen state
    let reference := composed a s q b (qtrials.map BoundQuantCall.invocation) qchosen.invocation
      owner callerDevice (calls.map BoundCall.invocation) chosen.invocation (abi.observe state)
    (actual.tensor,abi.observe actual.state,actual.currentDevice)=(reference.tensor,reference.memory,reference.currentDevice) := by
  unfold compiledComposed composed
  dsimp only
  rw [compiledWrapper_refines,compiledQuantWrapper_refines]

#axiomsClean compiledGrid_refines
#axiomsClean compiledRuntime_refines
#axiomsClean compiledWrapper_refines
#axiomsClean compiledQuantWrapper_refines
#axiomsClean compiledComposed_refines
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization
