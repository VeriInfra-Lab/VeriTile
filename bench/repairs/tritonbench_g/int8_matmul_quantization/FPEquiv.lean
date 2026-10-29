import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.Public
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.DirectNativeRow
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.DirectHelperResults
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.DirectComposedNative

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization
noncomputable section
set_option maxHeartbeats 5000000

theorem fp_equiv (a : Ops (BitVec 32) Word32 Word16) (isa : FixedISAProfile) (scalar : ScalarProfile)
    (numeric : PrimitiveLeaves a isa scalar) (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a)
    (hook : ZeroHookABI abi) (s : Shape) (b : Buffers) (sep : b.Separate) (owner callerDevice : Nat)
    (calls : List (BoundCall (BitVec 32) Word32 Word16 s b)) (chosen : BoundCall (BitVec 32) Word32 Word16 s b)
    (state : σ) (x : MatrixCell s) :
    let result := matmul_int8 abi hook owner callerDevice calls chosen state
    result.tensor=outputTensor s b owner ∧ result.currentDevice=callerDevice ∧
    abi.observe result.state (outputAddress s b x)=.inr
      (wordReference (fpOps isa scalar) s chosen.invocation.config b chosen.invocation.selected chosen.invocation.schedule (abi.observe state) x) := by
  dsimp only [matmul_int8]
  have h := compiledWrapper_refines abi hook owner callerDevice calls chosen state
  have tensor := congrArg Prod.fst h
  have memory := congrArg (fun t => t.2.1) h
  have device := congrArg (fun t => t.2.2) h
  simp only [Prod.fst,Prod.snd] at tensor memory device
  have ret := wrapper_returns a s b owner callerDevice (calls.map BoundCall.invocation) chosen.invocation (abi.observe state)
  refine ⟨tensor.trans ret.1,device.trans ret.2,?_⟩
  rw [memory,wrapper_word a s b sep,←numeric.eq]

theorem fp_quantizer (a : Ops (BitVec 32) Word32 Word16) (isa : FixedISAProfile) (scalar : ScalarProfile)
    (numeric : PrimitiveLeaves a isa scalar) (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a)
    (q : QuantBuffers) (sep : q.Separate) (calls : List (BoundQuantCall (BitVec 32) Word32 Word16 G K q))
    (chosen : BoundQuantCall (BitVec 32) Word32 Word16 G K q) (state : σ) (x : QuantIndex G K) :
    abi.observe (quantize_int8_perrow abi calls chosen state) (q.writeAddress x)=
      quantExpected (fpOps isa scalar) q chosen.invocation.plan (abi.observe state) x := by
  rw [quantize_int8_perrow,compiledQuantWrapper_refines,quantWrapper_at a q sep,←numeric.eq]

theorem fp_composed (a : Ops (BitVec 32) Word32 Word16) (isa : FixedISAProfile) (scalar : ScalarProfile)
    (numeric : PrimitiveLeaves a isa scalar) (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a)
    (hook : ZeroHookABI abi) (s : Shape) (q : QuantBuffers) (qsep : q.Separate) (b : Buffers) (sep : b.Separate)
    (link : CompositionLink q b)
    (qtrials : List (BoundQuantCall (BitVec 32) Word32 Word16 s.M s.K q))
    (qchosen : BoundQuantCall (BitVec 32) Word32 Word16 s.M s.K q) (owner callerDevice : Nat)
    (calls : List (BoundCall (BitVec 32) Word32 Word16 s b)) (chosen : BoundCall (BitVec 32) Word32 Word16 s b)
    (state : σ) (x : MatrixCell s) :
    abi.observe (matmul_quantize_int8 abi hook link qtrials qchosen owner callerDevice calls chosen state).state (outputAddress s b x)=
      .inr (composedReference (fpOps isa scalar) s q b qchosen.invocation.plan chosen.invocation (abi.observe state) x) := by
  have h := congrArg (fun t => t.2.1) (compiledComposed_refines abi hook link qtrials qchosen owner callerDevice calls chosen state)
  simp only [Prod.fst,Prod.snd] at h
  rw [matmul_quantize_int8,h,composed_reference a s q qsep b sep,←numeric.eq]

/-- Torch uses its own scalar profile instance. In particular its propagating
amax is not identified with Triton's maxnum by this theorem. -/
theorem fp_helper (a : Ops (BitVec 32) Word32 Word16) (isa : FixedISAProfile) (torchScalar : ScalarProfile)
    (numeric : PrimitiveLeaves a isa torchScalar) (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a)
    (call : BoundHelper (BitVec 32) Word32 Word16 rows cols G K reductionAxis temporary final)
    (state : σ) (x : QuantCell G K) (g : Fin G) :
    abi.observe (quantize_int8 abi call state) (final.dataAddress x)=
      .inl (quantValue (fpOps isa torchScalar) temporary call.invocation.plan (abi.observe state) x) ∧
    abi.observe (quantize_int8 abi call state) (temporary.scaleAddress g)=
      .inr (.inl (quantScale (fpOps isa torchScalar) temporary call.invocation.plan (abi.observe state) g)) := by
  rw [quantize_int8,compiledHelper_refines]
  constructor
  · rw [helper_value a reductionAxis temporary final call.admitted,←numeric.eq]
  · rw [helper_scale a reductionAxis temporary final call.admitted,←numeric.eq]

theorem fp_frame (a : Ops (BitVec 32) Word32 Word16)
    (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a) (raw : RawByteABI abi) (hook : ZeroHookABI abi)
    (owner callerDevice : Nat) (calls : List (BoundCall (BitVec 32) Word32 Word16 s b))
    (chosen : BoundCall (BitVec 32) Word32 Word16 s b) (state : σ) (storage offset : Nat)
    (outside : ¬OutputRegion s b (storage,offset/dtypeBytes (raw.dtype storage))) :
    raw.bytes (matmul_int8 abi hook owner callerDevice calls chosen state).state (storage,offset)=raw.bytes state (storage,offset) := by
  have h := congrArg (fun t => t.2.1) (compiledWrapper_refines abi hook owner callerDevice calls chosen state)
  simp only [Prod.fst,Prod.snd] at h
  rw [raw.decode,raw.decode,matmul_int8,h,wrapper_full_frame a s b owner callerDevice _ _ _ _ outside]

theorem fp_output_bytes (a : Ops (BitVec 32) Word32 Word16) (isa : FixedISAProfile) (scalar : ScalarProfile)
    (numeric : PrimitiveLeaves a isa scalar) (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a)
    (raw : RawByteABI abi) (hook : ZeroHookABI abi) (s : Shape) (b : Buffers) (sep : b.Separate)
    (owner callerDevice : Nat) (calls : List (BoundCall (BitVec 32) Word32 Word16 s b))
    (chosen : BoundCall (BitVec 32) Word32 Word16 s b) (state : σ) (x : MatrixCell s)
    (outputKind : raw.dtype b.out=precisionDType b.outPrecision)
    (i : Fin (dtypeBytes (raw.dtype b.out))) (positive : 0<dtypeBytes (raw.dtype b.out)) :
    raw.bytes (matmul_int8 abi hook owner callerDevice calls chosen state).state
      (b.out,dtypeBytes (raw.dtype b.out)*(b.outOffset+x.1.val*s.N+x.2.val)+i.val)=
      cellByte (.inr (wordReference (fpOps isa scalar) s chosen.invocation.config b chosen.invocation.selected
        chosen.invocation.schedule (abi.observe state) x)) i.val := by
  have h := (fp_equiv a isa scalar numeric abi hook s b sep owner callerDevice calls chosen state x).2.2
  rw [raw.decode]
  simp only [Nat.mul_add_div positive,Nat.div_eq_of_lt i.isLt,Nat.add_zero,
    Nat.mul_add_mod,Nat.mul_mod_right,Nat.zero_add,Nat.mod_eq_of_lt i.isLt]
  change cellByte (abi.observe _ (outputAddress s b x)) i.val=_
  rw [h]

#axiomsClean fp_equiv
#axiomsClean fp_quantizer
#axiomsClean fp_composed
#axiomsClean fp_helper
#axiomsClean fp_frame
#axiomsClean fp_output_bytes

theorem fp_public_result (a : Ops (BitVec 32) Word32 Word16) (isa : FixedISAProfile) (scalar : ScalarProfile)
    (numeric : PrimitiveLeaves a isa scalar) (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a)
    (hook : ZeroHookABI abi) (s : Shape) (b : Buffers) (owner callerDevice : Nat)
    (A AS B BS returned : TensorHandle) (supplied : Option TensorHandle)
    (state : σ) (admit : GemmAdmission s b owner A AS B BS supplied returned (abi.observe state))
    (calls : List (BoundCall (BitVec 32) Word32 Word16 s b)) (chosen : BoundCall (BitVec 32) Word32 Word16 s b)
    (x : MatrixCell s) :
    (publicGemm returned abi hook owner callerDevice calls chosen state).1=returned ∧
    (∀ out,supplied=some out → returned=out) ∧
    abi.observe (publicGemm returned abi hook owner callerDevice calls chosen state).2.state (outputAddress s b x)=
      .inr (wordReference (fpOps isa scalar) s chosen.invocation.config b chosen.invocation.selected
        chosen.invocation.schedule (abi.observe state) x) := by
  exact ⟨rfl,admit.outputIdentity,
    (fp_equiv a isa scalar numeric abi hook s b admit.separate owner callerDevice calls chosen state x).2.2⟩

#axiomsClean fp_public_result
namespace FloatingSpecification
open Direct

/-- Whole checked native GEMM, original input words, exact supplied/fresh
object metadata, every output byte and all untouched caller bytes. -/
theorem matmul_int8_native_if_local (base : Nat → Nat) (r : GemmRequest) (laws : GemmLaws base r)
    (a : Ops (BitVec 32) Word32 Word16) (isa : FixedISAProfile) (scalar : ScalarProfile)
    (numeric : PrimitiveLeaves a isa scalar) (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a)
    (raw : RawByteABI abi) (hook : ZeroHookABI abi) (callerDevice : Nat) (caller : σ)
    (live : Nat → Prop) (inputs : r.live live) (inputWords : GemmInputABI r abi raw caller)
    (alloc : NativeGemmAllocation r abi caller live)
    (outputLaws : TensorLaws base alloc.output)
    (kind : raw.dtype alloc.output.info.storage=precisionDType (r.buffers alloc.output.info.storage).outPrecision)
    (trials : List (GemmSource (BitVec 32) Word32 Word16 r.shape (r.buffers alloc.output.info.storage)))
    (chosen : GemmSource (BitVec 32) Word32 Word16 r.shape (r.buffers alloc.output.info.storage)) :
    ∃ result,runGemmNative r abi hook callerDevice caller live alloc trials chosen=.ok result ∧
      result.tensor=alloc.output ∧ GemmDescriptor r result.tensor ∧ OutputViewBinding r result.tensor ∧ result.currentDevice=callerDevice ∧ (∀ t∈r.views,InputPointerBinding base t) ∧
      InputPointerBinding base result.tensor ∧
      (∀ cell : MatrixCell r.shape,∀ i : Fin (dtypeBytes (precisionDType (r.buffers alloc.output.info.storage).outPrecision)),
        raw.bytes result.state (alloc.output.info.storage,dtypeBytes (precisionDType (r.buffers alloc.output.info.storage).outPrecision)*
          ((r.buffers alloc.output.info.storage).outOffset+cell.1.val*r.shape.N+cell.2.val)+i.val)=
        cellByte (.inr (logicalWordReference (fpOps isa scalar) r.shape chosen.invocation.config
          (r.buffers alloc.output.info.storage).outPrecision chosen.invocation.selected chosen.invocation.schedule
          (originalGemmWords (fpOps isa scalar) r (abi.observe caller)) cell)) i.val) ∧
      (∀ storage offset,live storage →
        ¬OutputRegion r.shape (r.buffers alloc.output.info.storage) (storage,offset/dtypeBytes (raw.dtype storage)) →
        raw.bytes result.state (storage,offset)=raw.bytes caller (storage,offset)) := by
  obtain ⟨result,ok,identity,descriptor,readback,device,values,frame⟩:=gemm_native_complete r abi hook callerDevice caller live inputs alloc trials chosen
  rw [identity] at values frame
  refine ⟨result,ok,identity,descriptor,readback,device,(fun t ht=>input_pointer_binding base t (laws.inputs t ht)),
    (by rw [identity];exact input_pointer_binding base alloc.output outputLaws),?_,?_⟩
  · intro cell i
    have eq:=congrArg (fun ops=>wordReference ops r.shape chosen.invocation.config (r.buffers alloc.output.info.storage) chosen.invocation.selected chosen.invocation.schedule (abi.observe caller) cell) numeric.eq
    have h:=(values cell).trans (congrArg Sum.inr eq)
    dsimp only at h
    rw [gemm_word_original base r laws] at h
    rw [raw.decode,kind]
    simp only [Nat.mul_add_div (dtypeBytes_positive _),Nat.div_eq_of_lt i.isLt,Nat.add_zero,
      Nat.mul_add_mod,Nat.mul_mod_right,Nat.zero_add,Nat.mod_eq_of_lt i.isLt]
    change cellByte (abi.observe result.state (outputAddress r.shape (r.buffers alloc.output.info.storage) cell)) i.val=_
    rw [h]
  · intro storage offset owned outside
    rw [raw.decode,raw.decode,frame _ owned outside]

/-- Whole row producer with original FP32 input words, explicit INT8 data and
FP16 scales, exact returned pair metadata and pre-allocation caller bytes. -/
theorem quantize_int8_perrow_native_if_local (base : Nat → Nat) (r : RowRequest) (laws : TensorLaws base r.input)
    (a : Ops (BitVec 32) Word32 Word16) (isa : FixedISAProfile) (scalar : ScalarProfile)
    (numeric : PrimitiveLeaves a isa scalar) (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a)
    (raw : RawByteABI abi) (callerDevice : Nat) (caller : σ) (live : Nat → Prop)
    (input : live r.input.info.storage) (inputWords : RowInputABI r abi raw caller)
    (alloc : NativeRowAllocation r abi caller live)
    (dataLaws : TensorLaws base alloc.logical.data) (scaleLaws : TensorLaws base alloc.logical.scales)
    (dataKind : raw.dtype alloc.logical.data.info.storage=.int8)
    (scaleKind : raw.dtype alloc.logical.scales.info.storage=.fp16)
    (trials : List (RowSource (BitVec 32) Word32 Word16 r.G r.K (r.buffers alloc.logical.data.info.storage alloc.logical.scales.info.storage)))
    (chosen : RowSource (BitVec 32) Word32 Word16 r.G r.K (r.buffers alloc.logical.data.info.storage alloc.logical.scales.info.storage)) :
    ∃ result,runRowNative r abi callerDevice caller live alloc trials chosen=.ok result ∧
      FreshDescriptor result.data (.other "int8") [r.G,r.K] [max r.K 1,1] r.input.info.metadata.device ∧
      FreshDescriptor result.scales (.other "float16") [r.G] [1] r.input.info.metadata.device ∧
      RowViewBinding r result.data result.scales ∧ result.currentDevice=callerDevice ∧ InputPointerBinding base r.input ∧
      InputPointerBinding base result.data ∧ InputPointerBinding base result.scales ∧
      (∀ cell : QuantCell r.G r.K,raw.bytes result.state (result.data.info.storage,cell.1.val*r.K+cell.2.val)=
        rowMachineValue (fpOps isa scalar) r chosen.invocation.plan (abi.observe caller) cell) ∧
      (∀ g : Fin r.G,∀ i : Fin 2,raw.bytes result.state (result.scales.info.storage,2*g.val+i.val)=
        ((fpOps isa scalar).narrow (rowMachineScale (fpOps isa scalar) r chosen.invocation.plan (abi.observe caller) g)).bits.extractLsb' (8*i.val) 8) ∧
      (∀ storage offset,live storage → raw.bytes result.state (storage,offset)=raw.bytes caller (storage,offset)) := by
  obtain ⟨result,ok,data,scales,dataDesc,scaleDesc,readback,device,values,frame⟩:=row_native_complete r abi callerDevice caller live input alloc trials chosen
  have dk : raw.dtype result.data.info.storage=.int8:=by rw [data];exact dataKind
  have sk : raw.dtype result.scales.info.storage=.fp16:=by rw [scales];exact scaleKind
  refine ⟨result,ok,dataDesc,scaleDesc,readback,device,input_pointer_binding base r.input laws,
    (by rw [data];exact input_pointer_binding base alloc.logical.data dataLaws),
    (by rw [scales];exact input_pointer_binding base alloc.logical.scales scaleLaws),?_,?_,?_⟩
  · intro cell
    have eq:=congrArg (fun ops=>quantExpected ops (r.buffers result.data.info.storage result.scales.info.storage) chosen.invocation.plan (abi.observe caller) (.inl cell)) numeric.eq
    have h:=(values (.inl cell)).trans eq
    dsimp only at h
    rw [row_machine_expected_original base r laws] at h
    simp only [rowMachineExpected] at h
    have h' : abi.observe result.state (result.data.info.storage,cell.1.val*r.K+cell.2.val)=
        .inl (rowMachineValue (fpOps isa scalar) r chosen.invocation.plan (abi.observe caller) cell) := by
      simpa [RowRequest.buffers,rowBuffers,QuantBuffers.writeAddress,QuantBuffers.dataAddress,RowRequest.K] using h
    rw [raw.decode,dk]
    simp only [dtypeBytes,Nat.div_one,Nat.mod_one,h',cellByte]
  · intro g i
    have eq:=congrArg (fun ops=>quantExpected ops (r.buffers result.data.info.storage result.scales.info.storage) chosen.invocation.plan (abi.observe caller) (.inr g)) numeric.eq
    have h:=(values (.inr g)).trans eq
    dsimp only at h
    rw [row_machine_expected_original base r laws] at h
    simp only [rowMachineExpected] at h
    have h' : abi.observe result.state (result.scales.info.storage,g.val)=
        .inr (.inr ((fpOps isa scalar).narrow (rowMachineScale (fpOps isa scalar) r chosen.invocation.plan (abi.observe caller) g))) := by
      simpa [RowRequest.buffers,rowBuffers,QuantBuffers.writeAddress,QuantBuffers.scaleAddress] using h
    rw [raw.decode,sk]
    simp only [dtypeBytes,Nat.mul_add_div (by decide : 0<2),Nat.div_eq_of_lt i.isLt,Nat.add_zero,
      Nat.mul_add_mod,Nat.mul_mod_right,Nat.zero_add,Nat.mod_eq_of_lt i.isLt,h',cellByte]
  · intro storage offset owned
    rw [raw.decode,raw.decode,frame _ owned]

abbrev quantize_int8_native_if_local := @helper_native_if_local
abbrev matmul_quantize_int8_native_if_local := @composed_native_if_local
#axiomsClean matmul_quantize_int8_native_if_local
#axiomsClean quantize_int8_native_if_local
#axiomsClean matmul_int8_native_if_local
#axiomsClean quantize_int8_perrow_native_if_local
end FloatingSpecification

end
end VeriTile.Bench.Repairs.Int8MatmulQuantization
