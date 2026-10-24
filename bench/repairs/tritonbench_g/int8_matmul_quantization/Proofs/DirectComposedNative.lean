import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.DirectComposed
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.DirectNativeRow

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.Direct
noncomputable section
set_option maxHeartbeats 3000000

/-- Independent producer-to-GEMM word target on original inputs. The FP16
scale is narrowed then widened before GEMM, exactly as the two kernels do. -/
def composedWords (a : Ops I F H) (r : ComposedRequest) (plan : Fin r.row.G → ReductionPlan r.row.K)
    (m : Memory F H) : LogicalGemmWords F where
  A := fun row k=>if hr : row<r.shape.M then if hk : k<r.shape.K then
    rowMachineValue a r.row plan m (⟨row,hr⟩,⟨k,hk⟩) else 0 else 0
  B := fun k n=>read8 m (logicalAddress r.b (k*r.shape.N+n))
  aScale := fun row=>if hr : row<r.shape.M then
    a.widen (a.narrow (rowMachineScale a r.row plan m ⟨row,hr⟩)) else a.fzero
  bScale := fun n=>read32 a.toOps m (logicalAddress r.bScale n)

theorem composed_accumulation_original (base : Nat → Nat) (r : ComposedRequest) (laws : ComposedLaws base r)
    (a : Ops I F H) (caller : Memory F H) (live : Nat → Prop) (inputs : r.live live) (producer : RowAllocation r.row caller live)
    (qtrials : List (QuantInvocation r.row.G r.row.K)) (qchosen : QuantInvocation r.row.G r.row.K)
    (c : Config) (selected : Selection r.shape c) (out : Nat) (cell : MatrixCell r.shape) (z t : Nat) :
    accumulation a r.shape c ((producedRequest r caller live inputs producer).buffers out) selected
      (producerMemory r.row a caller live producer qtrials qchosen) cell.1 cell.2 z t=
    logicalAccumulation a r.shape c selected (composedWords a r qchosen.plan caller) cell.1 cell.2 z t := by
  let b:=(producedRequest r caller live inputs producer).buffers out
  let m:=producerMemory r.row a caller live producer qtrials qchosen
  have ha : ∀ t,operandA r.shape c b m cell.1 t z=
      (fun l=>if sourceK c t z l.val<r.shape.K then (composedWords a r qchosen.plan caller).A cell.1 (sourceK c t z l.val) else 0) := by
    intro t;funext l;unfold operandA;split_ifs with hk
    · have av:=composed_a_value r a caller live inputs producer qtrials qchosen out cell.1 ⟨_,hk⟩
      have expected:=row_machine_expected_original base r.row laws.fpa a producer.data.info.storage producer.scales.info.storage qchosen.plan caller (.inl (cell.1,⟨_,hk⟩))
      have qv:=Sum.inl.inj expected
      simpa only [composedWords,dif_pos cell.1.isLt,dif_pos hk] using av.trans qv
    · rfl
  have hb : ∀ t,operandB r.shape c b m cell.2 t z=
      (fun l=>if sourceK c t z l.val<r.shape.K then (composedWords a r qchosen.plan caller).B (sourceK c t z l.val) cell.2 else 0) := by
    intro t;funext l;unfold operandB;split_ifs with hk
    · exact composed_b_value base r laws a caller live inputs producer qtrials qchosen out ⟨_,hk⟩ cell.2
    · rfl
  change accumulation a r.shape c b selected m cell.1 cell.2 z t=_
  induction t with
  | zero=>rfl
  | succ t ih=>simp only [accumulation,logicalAccumulation,ha,hb,ih]
theorem composed_partial_original (base : Nat → Nat) (r : ComposedRequest) (laws : ComposedLaws base r)
    (a : Ops I F H) (caller : Memory F H) (live : Nat → Prop) (inputs : r.live live) (producer : RowAllocation r.row caller live)
    (qtrials : List (QuantInvocation r.row.G r.row.K)) (qchosen : QuantInvocation r.row.G r.row.K)
    (c : Config) (selected : Selection r.shape c) (out : Nat) (cell : MatrixCell r.shape) (z : Nat) :
    partialValue a r.shape c ((producedRequest r caller live inputs producer).buffers out) selected
      (producerMemory r.row a caller live producer qtrials qchosen) cell.1 cell.2 z=
    logicalPartial a r.shape c selected (composedWords a r qchosen.plan caller) cell.1 cell.2 z := by
  obtain ⟨asv,bsv⟩:=composed_scale_values base r laws a caller live inputs producer qtrials qchosen out cell
  rw [row_machine_scale_original base r.row laws.fpa] at asv
  unfold partialValue logicalPartial
  rw [composed_accumulation_original base r laws,asv,bsv]
  simp only [composedWords,dif_pos cell.1.isLt]
  rfl
theorem composed_word_original (base : Nat → Nat) (r : ComposedRequest) (laws : ComposedLaws base r)
    (a : Ops I F H) (caller : Memory F H) (live : Nat → Prop) (inputs : r.live live) (producer : RowAllocation r.row caller live)
    (qtrials : List (QuantInvocation r.row.G r.row.K)) (qchosen : QuantInvocation r.row.G r.row.K)
    (chosen : Invocation r.shape) (out : Nat) (cell : MatrixCell r.shape) :
    wordReference a r.shape chosen.config ((producedRequest r caller live inputs producer).buffers out) chosen.selected chosen.schedule
      (producerMemory r.row a caller live producer qtrials qchosen) cell=
    logicalWordReference a r.shape chosen.config (if singleOutput (r.out.map TensorView.info) then .single else .half)
      chosen.selected chosen.schedule (composedWords a r qchosen.plan caller) cell := by
  unfold wordReference logicalWordReference
  have h : (fun z : Fin chosen.config.S=>partialValue a r.shape chosen.config ((producedRequest r caller live inputs producer).buffers out) chosen.selected
      (producerMemory r.row a caller live producer qtrials qchosen) cell.1 cell.2 z)=
      (fun z : Fin chosen.config.S=>logicalPartial a r.shape chosen.config chosen.selected (composedWords a r qchosen.plan caller) cell.1 cell.2 z) := by
    funext z;exact composed_partial_original base r laws a caller live inputs producer qtrials qchosen chosen.config chosen.selected out cell z
  rw [h]
  rfl

def runComposedNative (r : ComposedRequest) (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi) (callerDevice : Nat)
    (caller : σ) (live : Nat → Prop) (inputs : r.live live) (producer : NativeRowAllocation r.row abi caller live)
    (qtrials : List (RowSource I F H r.row.G r.row.K (r.row.buffers producer.logical.data.info.storage producer.logical.scales.info.storage)))
    (qchosen : RowSource I F H r.row.G r.row.K (r.row.buffers producer.logical.data.info.storage producer.logical.scales.info.storage))
    (output : NativeGemmAllocation (producedRequest r (abi.observe caller) live inputs producer.logical) abi
      (rowNative r.row producer.logical.data.info.storage producer.logical.scales.info.storage abi qtrials qchosen producer.state)
      (extendedLive live producer.logical.data.info.storage producer.logical.scales.info.storage))
    (trials : List (GemmSource I F H r.shape ((producedRequest r (abi.observe caller) live inputs producer.logical).buffers output.output.info.storage)))
    (chosen : GemmSource I F H r.shape ((producedRequest r (abi.observe caller) live inputs producer.logical).buffers output.output.info.storage)) :
    Except InputError (NativeHostResult σ) := do
  checkComposedRequest r
  checkRow (.tensor r.fpa.info)
  runGemmNative (producedRequest r (abi.observe caller) live inputs producer.logical) abi hook callerDevice
    (rowNative r.row producer.logical.data.info.storage producer.logical.scales.info.storage abi qtrials qchosen producer.state) _ output trials chosen

theorem composed_native_complete (base : Nat → Nat) (r : ComposedRequest) (laws : ComposedLaws base r)
    (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi) (callerDevice : Nat)
    (caller : σ) (live : Nat → Prop) (inputs : r.live live) (producer : NativeRowAllocation r.row abi caller live)
    (qtrials : List (RowSource I F H r.row.G r.row.K (r.row.buffers producer.logical.data.info.storage producer.logical.scales.info.storage)))
    (qchosen : RowSource I F H r.row.G r.row.K (r.row.buffers producer.logical.data.info.storage producer.logical.scales.info.storage))
    (output : NativeGemmAllocation (producedRequest r (abi.observe caller) live inputs producer.logical) abi
      (rowNative r.row producer.logical.data.info.storage producer.logical.scales.info.storage abi qtrials qchosen producer.state)
      (extendedLive live producer.logical.data.info.storage producer.logical.scales.info.storage))
    (trials : List (GemmSource I F H r.shape ((producedRequest r (abi.observe caller) live inputs producer.logical).buffers output.output.info.storage)))
    (chosen : GemmSource I F H r.shape ((producedRequest r (abi.observe caller) live inputs producer.logical).buffers output.output.info.storage)) :
    ∃ result,runComposedNative r abi hook callerDevice caller live inputs producer qtrials qchosen output trials chosen=.ok result ∧
      result.tensor=output.output ∧ GemmDescriptor (producedRequest r (abi.observe caller) live inputs producer.logical) result.tensor ∧ OutputViewBinding (producedRequest r (abi.observe caller) live inputs producer.logical) result.tensor ∧ result.currentDevice=callerDevice ∧
      (∀ cell : MatrixCell r.shape,abi.observe result.state (outputAddress r.shape ((producedRequest r (abi.observe caller) live inputs producer.logical).buffers output.output.info.storage) cell)=
        .inr (logicalWordReference a r.shape chosen.invocation.config (if singleOutput (r.out.map TensorView.info) then .single else .half)
          chosen.invocation.selected chosen.invocation.schedule (composedWords a r qchosen.invocation.plan (abi.observe caller)) cell)) ∧
      (∀ p,live p.1 → ¬OutputRegion r.shape ((producedRequest r (abi.observe caller) live inputs producer.logical).buffers output.output.info.storage) p →
        abi.observe result.state p=abi.observe caller p) := by
  obtain ⟨result,ok,identity,descriptor,readback,device,values,frame⟩:=gemm_native_complete
    (producedRequest r (abi.observe caller) live inputs producer.logical) abi hook callerDevice _ _
      (produced_live r (abi.observe caller) live inputs producer.logical) output trials chosen
  rw [identity] at values frame
  rw [rowNative_refines,producer.observed] at values frame
  refine ⟨result,?_,identity,descriptor,readback,device,?_,?_⟩
  · simp [runComposedNative,checkComposedRequest_ok,(checkRow_iff (.tensor r.fpa.info)).mpr r.contract.1,Except.bind,bind,ok]
  · intro cell
    have h:=values cell
    change abi.observe result.state (outputAddress r.shape ((producedRequest r (abi.observe caller) live inputs producer.logical).buffers output.output.info.storage) cell)=
      Sum.inr (wordReference a r.shape chosen.invocation.config ((producedRequest r (abi.observe caller) live inputs producer.logical).buffers output.output.info.storage)
        chosen.invocation.selected chosen.invocation.schedule
        (producerMemory r.row a (abi.observe caller) live producer.logical (qtrials.map RowSource.invocation) qchosen.invocation) cell) at h
    exact h.trans (congrArg Sum.inr (composed_word_original base r laws a (abi.observe caller) live inputs producer.logical
      (qtrials.map RowSource.invocation) qchosen.invocation chosen.invocation output.output.info.storage cell))
  · intro p owned outside
    rw [frame p (Or.inl owned) outside]
    exact producer_frame r.row a (abi.observe caller) live (r.fpaLive live inputs) producer.logical
      (qtrials.map RowSource.invocation) qchosen.invocation p owned

#axiomsClean composed_word_original
#axiomsClean composed_native_complete
#specNonCircular composedWords avoiding [producerMemory,runComposedNative,quantWrapper,wordReference,quantValue,quantScale]

structure ComposedInputABI (r : ComposedRequest) (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a)
    (raw : RawByteABI abi) (caller : σ) : Prop where
  inputKind : raw.dtype r.fpa.info.storage=.fp32
  bKind : raw.dtype r.b.info.storage=.int8
  scaleKind : raw.dtype r.bScale.info.storage=.fp32
  inputWords : TensorWords .fp32 r.fpa (abi.observe caller)
  bWords : TensorWords .int8 r.b (abi.observe caller)
  scaleWords : TensorWords .fp32 r.bScale (abi.observe caller)

theorem composed_native_if_local (base : Nat → Nat) (r : ComposedRequest) (laws : ComposedLaws base r)
    (a : Ops (BitVec 32) Word32 Word16) (isa : FixedISAProfile) (scalar : ScalarProfile)
    (numeric : PrimitiveLeaves a isa scalar) (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a)
    (raw : RawByteABI abi) (hook : ZeroHookABI abi) (callerDevice : Nat)
    (caller : σ) (live : Nat → Prop) (inputs : r.live live) (inputWords : ComposedInputABI r abi raw caller)
    (producer : NativeRowAllocation r.row abi caller live)
    (producerDataLaws : TensorLaws base producer.logical.data)
    (producerScaleLaws : TensorLaws base producer.logical.scales)
    (producedDataKind : raw.dtype producer.logical.data.info.storage=.int8)
    (producedScaleKind : raw.dtype producer.logical.scales.info.storage=.fp16)
    (qtrials : List (RowSource (BitVec 32) Word32 Word16 r.row.G r.row.K (r.row.buffers producer.logical.data.info.storage producer.logical.scales.info.storage)))
    (qchosen : RowSource (BitVec 32) Word32 Word16 r.row.G r.row.K (r.row.buffers producer.logical.data.info.storage producer.logical.scales.info.storage))
    (output : NativeGemmAllocation (producedRequest r (abi.observe caller) live inputs producer.logical) abi
      (rowNative r.row producer.logical.data.info.storage producer.logical.scales.info.storage abi qtrials qchosen producer.state)
      (extendedLive live producer.logical.data.info.storage producer.logical.scales.info.storage))
    (outputLaws : TensorLaws base output.output)
    (kind : raw.dtype output.output.info.storage=precisionDType (if singleOutput (r.out.map TensorView.info) then .single else .half))
    (trials : List (GemmSource (BitVec 32) Word32 Word16 r.shape ((producedRequest r (abi.observe caller) live inputs producer.logical).buffers output.output.info.storage)))
    (chosen : GemmSource (BitVec 32) Word32 Word16 r.shape ((producedRequest r (abi.observe caller) live inputs producer.logical).buffers output.output.info.storage)) :
    ∃ result,runComposedNative r abi hook callerDevice caller live inputs producer qtrials qchosen output trials chosen=.ok result ∧
      result.tensor=output.output ∧ GemmDescriptor (producedRequest r (abi.observe caller) live inputs producer.logical) result.tensor ∧ OutputViewBinding (producedRequest r (abi.observe caller) live inputs producer.logical) result.tensor ∧ result.currentDevice=callerDevice ∧
      (InputPointerBinding base r.fpa ∧ InputPointerBinding base r.b ∧ InputPointerBinding base r.bScale) ∧
      InputPointerBinding base producer.logical.data ∧ InputPointerBinding base producer.logical.scales ∧
      InputPointerBinding base result.tensor ∧
      (∀ cell : MatrixCell r.shape,∀ i : Fin (dtypeBytes (precisionDType (if singleOutput (r.out.map TensorView.info) then .single else .half))),
        raw.bytes result.state (output.output.info.storage,dtypeBytes (precisionDType (if singleOutput (r.out.map TensorView.info) then .single else .half))*
          (offset (r.out.map TensorView.info)+cell.1.val*r.shape.N+cell.2.val)+i.val)=
        cellByte (.inr (logicalWordReference (fpOps isa scalar) r.shape chosen.invocation.config (if singleOutput (r.out.map TensorView.info) then .single else .half)
          chosen.invocation.selected chosen.invocation.schedule (composedWords (fpOps isa scalar) r qchosen.invocation.plan (abi.observe caller)) cell)) i.val) ∧
      (∀ storage byte,live storage →
        ¬OutputRegion r.shape ((producedRequest r (abi.observe caller) live inputs producer.logical).buffers output.output.info.storage)
          (storage,byte/dtypeBytes (raw.dtype storage)) → raw.bytes result.state (storage,byte)=raw.bytes caller (storage,byte)) := by
  obtain ⟨result,ok,identity,descriptor,readback,device,values,frame⟩:=composed_native_complete base r laws abi hook callerDevice caller live inputs producer qtrials qchosen output trials chosen
  refine ⟨result,ok,identity,descriptor,readback,device,⟨input_pointer_binding base r.fpa laws.fpa,input_pointer_binding base r.b laws.b,
    input_pointer_binding base r.bScale laws.bScale⟩,
    input_pointer_binding base producer.logical.data producerDataLaws,
    input_pointer_binding base producer.logical.scales producerScaleLaws,
    (by rw [identity];exact input_pointer_binding base output.output outputLaws),?_,?_⟩
  · intro cell i
    have eq:=congrArg (fun ops=>logicalWordReference ops r.shape chosen.invocation.config (if singleOutput (r.out.map TensorView.info) then .single else .half)
      chosen.invocation.selected chosen.invocation.schedule (composedWords ops r qchosen.invocation.plan (abi.observe caller)) cell) numeric.eq
    have h:=(values cell).trans (congrArg Sum.inr eq)
    dsimp only at h
    rw [raw.decode,kind]
    simp only [Nat.mul_add_div (dtypeBytes_positive _),Nat.div_eq_of_lt i.isLt,Nat.add_zero,
      Nat.mul_add_mod,Nat.mul_mod_right,Nat.zero_add,Nat.mod_eq_of_lt i.isLt]
    change cellByte (abi.observe result.state (outputAddress r.shape
      ((producedRequest r (abi.observe caller) live inputs producer.logical).buffers output.output.info.storage) cell)) i.val=_
    rw [h]
  · intro storage byte owned outside
    rw [raw.decode,raw.decode,frame _ owned outside]

#axiomsClean composed_native_if_local
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.Direct
