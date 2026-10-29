import bench.repairs.tritonbench_g.int8_dequant_matmul.Proofs.DirectNative
import bench.repairs.tritonbench_g.int8_dequant_matmul.Proofs.DirectWords
import bench.repairs.tritonbench_g.int8_dequant_matmul.Proofs.Ingress

namespace VeriTile.Bench.Repairs.Int8DequantMatmul
noncomputable section
set_option maxHeartbeats 4000000

/-- Exact native FP16 output words follow only from scalar/fixed K32 numeric
leaves and ordinary source/compiler/memory/Tensor contracts. This theorem
consumes the same compiled code carried by every bound autotune invocation. -/
theorem fp_equiv (a : Ops (BitVec 32) Word32 Word16) (profile : FixedISAProfile)
    (numeric : PrimitiveLeaves a profile) (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a)
    (hook : ZeroHookABI abi) (s : Shape) (b : Buffers) (fresh : b.Fresh) (hasBias : Bool)
    (owner callerDevice : Nat)
    (benchmarks : List (BoundCall (BitVec 32) Word32 Word16 s b Source.runtimeDivisor hasBias))
    (chosen : BoundCall (BitVec 32) Word32 Word16 s b Source.runtimeDivisor hasBias)
    (state : σ) (x : MatrixCell s) :
    let returned := compiledWrapper abi hook owner callerDevice benchmarks chosen state
    returned.tensor=outputTensor s b owner ∧ returned.currentDevice=callerDevice ∧
    abi.observe returned.state (outputAddress s b x)=
      .inr (.inr (wordReference (fpOps profile) s chosen.invocation.config b chosen.invocation.selected
        Source.runtimeDivisor hasBias chosen.invocation.schedule (abi.observe state) x)) ∧
    InputsAgree b (abi.observe returned.state) (abi.observe state) := by
  dsimp only
  have result := compiledWrapper_refines abi hook owner callerDevice benchmarks chosen state
  have tensor := congrArg Prod.fst result
  have memory := congrArg (fun t => t.2.1) result
  have device := congrArg (fun t => t.2.2) result
  simp only [Prod.fst,Prod.snd] at tensor memory device
  obtain ⟨ret,dev⟩ := wrapper_returns a s b Source.runtimeDivisor hasBias owner callerDevice
    (benchmarks.map BoundCall.invocation) chosen.invocation (abi.observe state)
  refine ⟨tensor.trans ret,device.trans dev,?_,?_⟩
  · rw [memory,wrapper_word _ _ _ fresh]
    have eq := congrArg (fun ops => wordReference ops s chosen.invocation.config b chosen.invocation.selected
      Source.runtimeDivisor hasBias chosen.invocation.schedule (abi.observe state) x) numeric.eq
    exact congrArg (fun value : Word16 => (Sum.inr (Sum.inr value) : Sum (BitVec 8) (Sum Word32 Word16))) eq
  · rw [memory]
    exact wrapper_frame _ _ _ _ _ _ _ _ _ _

theorem fp_caller_bytes (a : Ops (BitVec 32) Word32 Word16)
    (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a) (raw : RawByteABI abi)
    (hook : ZeroHookABI abi) (s : Shape) (b : Buffers) (hasBias : Bool) (owner callerDevice : Nat)
    (benchmarks : List (BoundCall (BitVec 32) Word32 Word16 s b Source.runtimeDivisor hasBias))
    (chosen : BoundCall (BitVec 32) Word32 Word16 s b Source.runtimeDivisor hasBias)
    (state : σ) (storage offset : Nat) (caller : storage≠b.out) :
    raw.bytes (compiledWrapper abi hook owner callerDevice benchmarks chosen state).state (storage,offset)=
      raw.bytes state (storage,offset) := by
  apply raw.frame abi b state _ _ storage offset caller
  have result := compiledWrapper_refines abi hook owner callerDevice benchmarks chosen state
  have memory := congrArg (fun t => t.2.1) result
  simp only [Prod.fst,Prod.snd] at memory
  rw [memory]
  exact wrapper_frame _ _ _ _ _ _ _ _ _ _

/-- The two bytes of every physical FP16 result, including arbitrary output
storage offsets, are those of the independent ordered machine reference. -/
theorem fp_output_bytes (a : Ops (BitVec 32) Word32 Word16) (profile : FixedISAProfile)
    (numeric : PrimitiveLeaves a profile) (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a)
    (raw : RawByteABI abi) (hook : ZeroHookABI abi) (s : Shape) (b : Buffers) (fresh : b.Fresh)
    (hasBias : Bool) (owner callerDevice : Nat)
    (benchmarks : List (BoundCall (BitVec 32) Word32 Word16 s b Source.runtimeDivisor hasBias))
    (chosen : BoundCall (BitVec 32) Word32 Word16 s b Source.runtimeDivisor hasBias)
    (state : σ) (x : MatrixCell s) (i : Fin 2) (halfStorage : raw.dtype b.out=.fp16) :
    raw.bytes (compiledWrapper abi hook owner callerDevice benchmarks chosen state).state
      (b.out,2*(b.outOffset+x.1.val*s.N+x.2.val)+i.val)=
      (wordReference (fpOps profile) s chosen.invocation.config b chosen.invocation.selected
        Source.runtimeDivisor hasBias chosen.invocation.schedule (abi.observe state) x).bits.extractLsb' (8*i.val) 8 := by
  have word := (fp_equiv a profile numeric abi hook s b fresh hasBias owner callerDevice benchmarks chosen state x).2.2.1
  rw [raw.decode,halfStorage]
  simp only [DType.bytes,Nat.mul_add_div (by decide : 0<2),Nat.div_eq_of_lt i.isLt,Nat.add_zero,
    Nat.mul_add_mod,Nat.mul_mod_right,Nat.zero_add,Nat.mod_eq_of_lt i.isLt]
  change cellByte (abi.observe _ (outputAddress s b x)) i.val=_
  rw [word]
  rfl

theorem fp_all_caller_bytes (a : Ops (BitVec 32) Word32 Word16)
    (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a) (raw : RawByteABI abi)
    (hook : ZeroHookABI abi) (s : Shape) (b : Buffers) (hasBias : Bool) (owner callerDevice : Nat)
    (benchmarks : List (BoundCall (BitVec 32) Word32 Word16 s b Source.runtimeDivisor hasBias))
    (chosen : BoundCall (BitVec 32) Word32 Word16 s b Source.runtimeDivisor hasBias)
    (caller prepared : σ) (callerStorage : Nat → Prop)
    (allocation : AllocationABI b (abi.observe caller) (abi.observe prepared) callerStorage)
    (storage offset : Nat) (owned : callerStorage storage) :
    raw.bytes (compiledWrapper abi hook owner callerDevice benchmarks chosen prepared).state (storage,offset)=
      raw.bytes caller (storage,offset) := by
  have fresh : storage≠b.out := by intro h; rw [h] at owned; exact allocation.freshCaller owned
  rw [fp_caller_bytes a abi raw hook s b hasBias owner callerDevice benchmarks chosen prepared storage offset fresh,
    raw.decode,raw.decode,allocation.frame _ owned]

theorem fp_public_result (a : Ops (BitVec 32) Word32 Word16) (profile : FixedISAProfile)
    (numeric : PrimitiveLeaves a profile) (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a)
    (raw : RawByteABI abi) (hook : ZeroHookABI abi) (s : Shape) (b : Buffers)
    (hasBias : Bool) (owner callerDevice : Nat)
    (benchmarks : List (BoundCall (BitVec 32) Word32 Word16 s b Source.runtimeDivisor hasBias))
    (chosen : BoundCall (BitVec 32) Word32 Word16 s b Source.runtimeDivisor hasBias)
    (state : σ) (input : PublicInput abi raw s b hasBias state) (x : MatrixCell s) (i : Fin 2) :
    raw.bytes (compiledWrapper abi hook owner callerDevice benchmarks chosen state).state
      (b.out,2*(b.outOffset+x.1.val*s.N+x.2.val)+i.val)=
      (wordReference (fpOps profile) s chosen.invocation.config b chosen.invocation.selected
        Source.runtimeDivisor hasBias chosen.invocation.schedule (abi.observe state) x).bits.extractLsb' (8*i.val) 8 :=
  fp_output_bytes a profile numeric abi raw hook s b input.fresh hasBias owner callerDevice
    benchmarks chosen state x i input.outKind

namespace FloatingSpecification
open Direct

/-- Full typed wrapper with the native INT32/FP32/FP16 operation sequence.
No real-number homomorphism and no overflow, finite-scale or positivity gate. -/
theorem int8_dequant_matmul_wrapper_fp32 (r : Request) (a : Ops (BitVec 32) Word32 Word16)
    (callerDevice : Nat) (benchmarks : List (Invocation (requestShape r))) (chosen : Invocation (requestShape r))
    (caller : Memory Word32 Word16) (live : Nat → Prop) (inputs : InputsLive r live) (alloc : Allocation r caller live) :
    ∃ result, run r a Source.runtimeDivisor callerDevice benchmarks chosen caller live alloc=.ok result ∧
      result.tensor=outputTensor (requestShape r) (requestBuffers r alloc.storage) r.a.info.metadata.device.index ∧
      result.currentDevice=callerDevice ∧
      (∀ cell : MatrixCell (requestShape r),result.memory (outputAddress (requestShape r) (requestBuffers r alloc.storage) cell)=
        .inr (.inr (wordReference a (requestShape r) chosen.config (requestBuffers r alloc.storage) chosen.selected Source.runtimeDivisor
          (hasBias r) chosen.schedule caller cell))) ∧
      (∀ p,live p.1 → result.memory p=caller p) :=
  run_complete r a Source.runtimeDivisor callerDevice benchmarks chosen caller live inputs alloc

/-- Actual compiler-bound native wrapper, both returned FP16 bytes and every
pre-allocation caller byte. Numeric assumptions are fixed-instruction/scalar
primitive laws. Index/grid obligations are derived inside SourceCall.bound. -/
theorem native_buffers_if_local (r : Request)
    (a : Ops (BitVec 32) Word32 Word16) (profile : FixedISAProfile) (numeric : PrimitiveLeaves a profile)
    (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a) (raw : RawByteABI abi) (hook : ZeroHookABI abi)
    (callerDevice : Nat) (caller : σ) (live : Nat → Prop) (inputs : InputsLive r live)
    (alloc : NativeAllocation r abi caller live)
    (input : PublicInput abi raw (requestShape r) (requestBuffers r alloc.storage) (hasBias r) caller)
    (benchmarks : List (SourceCall (BitVec 32) Word32 Word16 r alloc.storage Source.runtimeDivisor))
    (chosen : SourceCall (BitVec 32) Word32 Word16 r alloc.storage Source.runtimeDivisor) :
    ∃ result, checkedNative r abi hook Source.runtimeDivisor callerDevice caller live alloc benchmarks chosen=.ok result ∧
      result.tensor=outputTensor (requestShape r) (requestBuffers r alloc.storage) r.a.info.metadata.device.index ∧
      result.currentDevice=callerDevice ∧
      (∀ cell : MatrixCell (requestShape r),∀ i : Fin 2,
        raw.bytes result.state (alloc.storage,2*(cell.1.val*(requestShape r).N+cell.2.val)+i.val)=
          (wordReference (fpOps profile) (requestShape r) chosen.invocation.config (requestBuffers r alloc.storage)
            chosen.invocation.selected Source.runtimeDivisor (hasBias r) chosen.invocation.schedule (abi.observe caller) cell).bits.extractLsb' (8*i.val) 8) ∧
      (∀ storage offset,live storage → raw.bytes result.state (storage,offset)=raw.bytes caller (storage,offset)) := by
  obtain ⟨result,ok,tensor,device,values,frame⟩:=native_complete r abi hook Source.runtimeDivisor callerDevice caller live inputs alloc benchmarks chosen
  refine ⟨result,ok,tensor,device,?_,?_⟩
  · intro cell i
    have value:=values cell
    have eq:=congrArg (fun ops=>wordReference ops (requestShape r) chosen.invocation.config
      (requestBuffers r alloc.storage) chosen.invocation.selected Source.runtimeDivisor
      (hasBias r) chosen.invocation.schedule (abi.observe caller) cell) numeric.eq
    have valueFP := value.trans (congrArg
      (fun word : Word16=>(Sum.inr (Sum.inr word) : Sum (BitVec 8) (Sum Word32 Word16))) eq)
    have kind : raw.dtype alloc.storage=.fp16:=input.outKind
    rw [raw.decode,kind]
    simp only [DType.bytes,Nat.mul_add_div (by decide : 0<2),Nat.div_eq_of_lt i.isLt,Nat.add_zero,
      Nat.mul_add_mod,Nat.mul_mod_right,Nat.zero_add,Nat.mod_eq_of_lt i.isLt]
    simp only [outputAddress,requestBuffers,buffers,Nat.zero_add] at valueFP
    rw [valueFP]
    rfl
  · intro storage offset owned
    rw [raw.decode,raw.decode,frame _ owned]

/-- The full native result is now expressed on the original TensorViews.
Ordinary Tensor laws are consumed to derive both actual data_ptr addressing
and the independent logical-word reference; no detached descriptor bridge is
required from the caller. -/
theorem int8_dequant_matmul_native_if_local (base : Nat → Nat) (r : Request)
    (laws : RequestLaws base r)
    (a : Ops (BitVec 32) Word32 Word16) (profile : FixedISAProfile) (numeric : PrimitiveLeaves a profile)
    (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a) (raw : RawByteABI abi) (hook : ZeroHookABI abi)
    (callerDevice : Nat) (caller : σ) (live : Nat → Prop) (inputs : InputsLive r live)
    (alloc : NativeAllocation r abi caller live)
    (input : PublicInput abi raw (requestShape r) (requestBuffers r alloc.storage) (hasBias r) caller)
    (benchmarks : List (SourceCall (BitVec 32) Word32 Word16 r alloc.storage Source.runtimeDivisor))
    (chosen : SourceCall (BitVec 32) Word32 Word16 r alloc.storage Source.runtimeDivisor) :
    ∃ result, checkedNative r abi hook Source.runtimeDivisor callerDevice caller live alloc benchmarks chosen=.ok result ∧
      result.tensor=outputTensor (requestShape r) (requestBuffers r alloc.storage) r.a.info.metadata.device.index ∧
      result.currentDevice=callerDevice ∧ OperandPointers base r ∧
      (∀ cell : MatrixCell (requestShape r),∀ i : Fin 2,
        raw.bytes result.state (alloc.storage,2*(cell.1.val*(requestShape r).N+cell.2.val)+i.val)=
          (logicalWordReference (fpOps profile) (requestShape r) chosen.invocation.config
            chosen.invocation.selected Source.runtimeDivisor chosen.invocation.schedule
            (originalWords (fpOps profile) r (abi.observe caller)) cell).bits.extractLsb' (8*i.val) 8) ∧
      (∀ storage offset,live storage → raw.bytes result.state (storage,offset)=raw.bytes caller (storage,offset)) := by
  obtain ⟨result,ok,tensor,device,values,frame⟩:=native_buffers_if_local r a profile numeric abi raw hook
    callerDevice caller live inputs alloc input benchmarks chosen
  refine ⟨result,ok,tensor,device,operand_pointers base r laws,?_,frame⟩
  intro cell i
  rw [values cell i,word_original base r laws]

#axiomsClean int8_dequant_matmul_wrapper_fp32
#axiomsClean int8_dequant_matmul_native_if_local
end FloatingSpecification

#axiomsClean fp_equiv
#axiomsClean fp_caller_bytes
#axiomsClean fp_output_bytes
#axiomsClean fp_all_caller_bytes
#axiomsClean fp_public_result
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Int8DequantMatmul
