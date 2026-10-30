import bench.tritonbench_g.int8_dequant_matmul.Proofs.Source
import bench.tritonbench_g.int8_dequant_matmul.Proofs.Coverage

namespace VeriTile.Bench.Repairs.Int8DequantMatmul
noncomputable section
set_option maxHeartbeats 4000000

/-- Tensor.zero_ is an ordinary fixed-zero fill contract, separate from all
numeric leaves. Each split>1 invocation uses it, including autotune trials and
the final selected invocation. -/
structure ZeroHookABI (abi : LocalEventABI I F H σ a) where
  zero : Shape → Buffers → σ → σ
  law : ∀ s b state,abi.observe (zero s b state)=reset a s b (abi.observe state)

def nativeInvoke (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (s : Shape) (b : Buffers) (divisor : F) (hasBias : Bool) (v : Invocation s) (state : σ) : σ :=
  let prepared := if v.config.S=1 then state else hook.zero s b state
  ((gridCode s v.config b v.selected divisor hasBias v.schedule.events).native abi prepared).2

theorem nativeInvoke_refines (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (s : Shape) (b : Buffers) (divisor : F) (hasBias : Bool) (v : Invocation s) (state : σ) :
    abi.observe (nativeInvoke abi hook s b divisor hasBias v state)=
      invoke a s b divisor hasBias v (abi.observe state) := by
  have result := congrArg Prod.snd ((gridCode s v.config b v.selected divisor hasBias v.schedule.events).native_refines
    abi (if v.config.S=1 then state else hook.zero s b state))
  simp only [Prod.snd,eval_gridCode] at result
  unfold nativeInvoke invoke launch
  rw [result]
  split_ifs <;> simp only [hook.law]

def nativeTrials (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (s : Shape) (b : Buffers) (divisor : F) (hasBias : Bool) : List (Invocation s) → σ → σ
  | [],state => state
  | v::vs,state => nativeTrials abi hook s b divisor hasBias vs (nativeInvoke abi hook s b divisor hasBias v state)
theorem nativeTrials_refines (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (s : Shape) (b : Buffers) (divisor : F) (hasBias : Bool) (vs : List (Invocation s)) (state : σ) :
    abi.observe (nativeTrials abi hook s b divisor hasBias vs state)=
      trials a s b divisor hasBias vs (abi.observe state) := by
  induction vs generalizing state with
  | nil => rfl
  | cons v vs ih => rw [nativeTrials,ih,nativeInvoke_refines,trials]

structure NativeReturned (σ : Type) where
  tensor : Tensor
  state : σ
  currentDevice : Nat
def nativeWrapper (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (s : Shape) (b : Buffers) (divisor : F) (hasBias : Bool) (owner callerDevice : Nat)
    (benchmarks : List (Invocation s)) (chosen : Invocation s) (state : σ) : NativeReturned σ :=
  if s.M=0 ∨ s.N=0 then ⟨outputTensor s b owner,state,callerDevice⟩
  else
    let invoked := DequantizeMatmul.deviceCall owner
      (fun entered => ⟨entered.currentDevice,nativeInvoke abi hook s b divisor hasBias chosen
        (nativeTrials abi hook s b divisor hasBias benchmarks entered.payload)⟩)
      ⟨callerDevice,state⟩
    ⟨outputTensor s b owner,invoked.payload,invoked.currentDevice⟩
theorem nativeWrapper_refines (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (s : Shape) (b : Buffers) (divisor : F) (hasBias : Bool) (owner callerDevice : Nat)
    (benchmarks : List (Invocation s)) (chosen : Invocation s) (state : σ) :
    let result := nativeWrapper abi hook s b divisor hasBias owner callerDevice benchmarks chosen state
    (result.tensor,abi.observe result.state,result.currentDevice)=
      let ref := wrapper a s b divisor hasBias owner callerDevice benchmarks chosen (abi.observe state)
      (ref.tensor,ref.memory,ref.currentDevice) := by
  unfold nativeWrapper wrapper
  split_ifs
  · rfl
  · simp only [DequantizeMatmul.deviceCall,nativeInvoke_refines,nativeTrials_refines]

/-- This binds the same production AST and typed JIT to the actual compiler's
selected local-event program. It is explicitly an ordinary frontend/compiler
implementation boundary, not a theorem that this repository verifies Triton.
The right-hand side is fixed from source loop coordinates, exact prior seed,
operand loads, selected fixed-instruction plans, epilogue, and commit order.
No final output, memory equality, numeric property, or frame is a field. -/
structure CompilerBinding {I H : Type} (s : Shape) (v : Invocation s) (b : Buffers)
    (da : Dense2 s.M s.K) (db : Dense2 s.K s.N) (dc : Dense2 s.M s.N)
    (divisor : F) (hasBias : Bool) where
  parsedProduction : String
  productionIdentity : parsedProduction=Source.productionAST
  jit : VeriTile.Triton.ComputeKernel
  jitIdentity : jit=Source.typedJIT s v.config da db dc hasBias
  admitted : Source.Admitted v.config
  addressABI : IndexABI s v.config b da db dc
  gridABI : GridABI s v.config
  dispatch : v.schedule.events.Perm (allCommits s v.config)
  lowered : Code I F H Unit
  sourceLowering : lowered=gridCode s v.config b v.selected divisor hasBias v.schedule.events

theorem CompilerBinding.schedule_complete
    (binding : CompilerBinding (I:=I) (H:=H) s v b da db dc divisor hasBias) (x : MatrixCell s) :
    (projection x v.schedule.events).Perm (List.finRange v.config.S) :=
  (Schedule.ofPermutation s v.config v.schedule.events binding.dispatch).complete x

theorem boundCompiler_grid (abi : LocalEventABI I F H σ a) (s : Shape) (v : Invocation s) (b : Buffers)
    (da : Dense2 s.M s.K) (db : Dense2 s.K s.N) (dc : Dense2 s.M s.N) (divisor : F) (hasBias : Bool)
    (binding : CompilerBinding (I:=I) (H:=H) s v b da db dc divisor hasBias) (state : σ) :
    abi.observe (binding.lowered.native abi state).2=
      gridRun a s v.config b v.selected divisor hasBias v.schedule.events (abi.observe state) := by
  rw [binding.sourceLowering]
  have h := congrArg Prod.snd ((gridCode s v.config b v.selected divisor hasBias v.schedule.events).native_refines abi state)
  simpa only [Prod.snd,eval_gridCode] using h

def DType.bytes : DType → Nat
  | .int8 => 1
  | .fp32 => 4
  | .fp16 => 2
def cellByte (slot : Sum (BitVec 8) (Sum Word32 Word16)) (i : Nat) : BitVec 8 :=
  match slot with
  | .inl byte => byte
  | .inr (.inl word) => word.bits.extractLsb' (8*i) 8
  | .inr (.inr word) => word.bits.extractLsb' (8*i) 8

/-- Raw storage representation is local and covers every caller byte,
including bytes outside all logical tensor views. No cross-dtype aliasing is
admitted: each storage allocation has one fixed element width. -/
structure RawByteABI (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a) where
  dtype : Nat → DType
  bytes : σ → Address → BitVec 8
  decode : ∀ state storage offset,
    bytes state (storage,offset)=cellByte
      (abi.observe state (storage,offset/(dtype storage).bytes)) (offset%(dtype storage).bytes)

theorem RawByteABI.frame (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a)
    (raw : RawByteABI abi) (b : Buffers) (before after : σ)
    (h : InputsAgree b (abi.observe after) (abi.observe before))
    (storage offset : Nat) (caller : storage≠b.out) :
    raw.bytes after (storage,offset)=raw.bytes before (storage,offset) := by
  rw [raw.decode,raw.decode,h _ caller]

#axiomsClean nativeInvoke_refines
#axiomsClean nativeTrials_refines
#axiomsClean nativeWrapper_refines
#axiomsClean boundCompiler_grid
#axiomsClean CompilerBinding.schedule_complete
#axiomsClean RawByteABI.frame
end
end VeriTile.Bench.Repairs.Int8DequantMatmul
