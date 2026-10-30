import bench.tritonbench_g.int8_dequant_matmul.Proofs.Native

namespace VeriTile.Bench.Repairs.Int8DequantMatmul
noncomputable section
set_option maxHeartbeats 4000000

/-- The compiler binding is consumed by actual execution, rather than merely
listed beside a theorem about an unrelated hand-selected reference graph. -/
structure BoundCall (I F H : Type) (s : Shape) (b : Buffers) (divisor : F) (hasBias : Bool) where
  invocation : Invocation s
  aDense : Dense2 s.M s.K
  bDense : Dense2 s.K s.N
  cDense : Dense2 s.M s.N
  outputStrides : cDense.rowStride=s.N ∧ cDense.colStride=1
  compiler : CompilerBinding (I:=I) (H:=H) s invocation b aDense bDense cDense divisor hasBias

def compiledInvoke (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (call : BoundCall I F H s b divisor hasBias) (state : σ) : σ :=
  let prepared := if call.invocation.config.S=1 then state else hook.zero s b state
  (call.compiler.lowered.native abi prepared).2
theorem compiledInvoke_eq (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (call : BoundCall I F H s b divisor hasBias) (state : σ) :
    compiledInvoke abi hook call state=nativeInvoke abi hook s b divisor hasBias call.invocation state := by
  simp only [compiledInvoke,nativeInvoke,call.compiler.sourceLowering]

def compiledTrials (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi) :
    List (BoundCall I F H s b divisor hasBias) → σ → σ
  | [],state => state
  | c::cs,state => compiledTrials abi hook cs (compiledInvoke abi hook c state)
theorem compiledTrials_eq (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (calls : List (BoundCall I F H s b divisor hasBias)) (state : σ) :
    compiledTrials abi hook calls state=
      nativeTrials abi hook s b divisor hasBias (calls.map BoundCall.invocation) state := by
  induction calls generalizing state with
  | nil => rfl
  | cons c cs ih => simp only [compiledTrials,compiledInvoke_eq,ih,List.map_cons,nativeTrials]

def compiledWrapper (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (owner callerDevice : Nat) (benchmarks : List (BoundCall I F H s b divisor hasBias))
    (chosen : BoundCall I F H s b divisor hasBias) (state : σ) : NativeReturned σ :=
  if s.M=0 ∨ s.N=0 then ⟨outputTensor s b owner,state,callerDevice⟩
  else
    let context := DequantizeMatmul.deviceCall owner
      (fun entered => ⟨entered.currentDevice,compiledInvoke abi hook chosen
        (compiledTrials abi hook benchmarks entered.payload)⟩) ⟨callerDevice,state⟩
    ⟨outputTensor s b owner,context.payload,context.currentDevice⟩
theorem compiledWrapper_eq (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (owner callerDevice : Nat) (benchmarks : List (BoundCall I F H s b divisor hasBias))
    (chosen : BoundCall I F H s b divisor hasBias) (state : σ) :
    compiledWrapper abi hook owner callerDevice benchmarks chosen state=
      nativeWrapper abi hook s b divisor hasBias owner callerDevice
        (benchmarks.map BoundCall.invocation) chosen.invocation state := by
  simp only [compiledWrapper,nativeWrapper,compiledInvoke_eq,compiledTrials_eq]

theorem compiledWrapper_refines (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (owner callerDevice : Nat) (benchmarks : List (BoundCall I F H s b divisor hasBias))
    (chosen : BoundCall I F H s b divisor hasBias) (state : σ) :
    let actual := compiledWrapper abi hook owner callerDevice benchmarks chosen state
    let reference := wrapper a s b divisor hasBias owner callerDevice
      (benchmarks.map BoundCall.invocation) chosen.invocation (abi.observe state)
    (actual.tensor,abi.observe actual.state,actual.currentDevice)=
      (reference.tensor,reference.memory,reference.currentDevice) := by
  rw [compiledWrapper_eq]
  exact nativeWrapper_refines _ _ _ _ _ _ _ _ _ _ _

#axiomsClean compiledInvoke_eq
#axiomsClean compiledTrials_eq
#axiomsClean compiledWrapper_eq
#axiomsClean compiledWrapper_refines
end
end VeriTile.Bench.Repairs.Int8DequantMatmul
