import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.DirectHelper
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.DirectNativeRow

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.Direct
noncomputable section
set_option maxHeartbeats 2000000

theorem helper_real_complete (base : Nat → Nat) (r : HelperRequest) (laws : TensorLaws base r.weight)
    (callerDevice : Nat) (chosen : QuantInvocation r.G r.K) (caller : Memory ℝ ℝ) (live : Nat → Prop)
    (input : live r.weight.info.storage) (alloc : HelperAllocation r caller live) :
    ∃ result,runHelper r realOps callerDevice chosen caller live alloc=.ok result ∧
      result.values=alloc.final ∧ result.scales=alloc.scales ∧ result.third=none ∧ result.currentDevice=callerDevice ∧
      (r.needsCopy=Bool.true → ColumnDescriptor r result.values) ∧
      (r.needsCopy=Bool.false → result.values=alloc.temporary) ∧
      FreshDescriptor result.scales (.floating .fp32) [r.G] [1] r.weight.info.metadata.device ∧
      HelperViewBinding r result.values result.scales ∧
      (∀ cell : QuantCell r.G r.K,result.memory (alloc.dest.dataAddress cell)=.inl (helperValueTarget r caller cell.1 cell.2)) ∧
      (∀ g : Fin r.G,result.memory (alloc.q.scaleAddress g)=.inr (.inl (helperScaleTarget r caller g))) ∧
      (∀ p,live p.1 → result.memory p=caller p) := by
  obtain ⟨result,ok,valuesID,scalesID,none,device,copy,nocopy,scales,readback,values,scaleValues,frame⟩:=helper_complete r realOps callerDevice chosen caller live input alloc
  refine ⟨result,ok,valuesID,scalesID,none,device,copy,nocopy,scales,readback,?_,?_,frame⟩
  · intro cell
    rw [values cell]
    change Sum.inl (quantValue realOps (r.temporary alloc.temporary.info.storage alloc.scales.info.storage) chosen.plan caller cell)=_
    rw [helper_value_original base r laws]
  · intro g
    rw [scaleValues g]
    change Sum.inr (Sum.inl (quantScale realOps (r.temporary alloc.temporary.info.storage alloc.scales.info.storage) chosen.plan caller g))=_
    rw [helper_scale_original base r laws]

def helperMachineScale (a : Ops I F H) (r : HelperRequest) (plan : Fin r.G → ReductionPlan r.K)
    (m : Memory F H) (g : Fin r.G) : F :=
  a.scale127 ((plan g).tree.eval a (fun k=>helperOriginal a r m g k))
def helperMachineValue (a : Ops I F H) (r : HelperRequest) (plan : Fin r.G → ReductionPlan r.K)
    (m : Memory F H) (cell : QuantCell r.G r.K) : Byte :=
  a.integer8 (a.divide (helperOriginal a r m cell.1 cell.2) (helperMachineScale a r plan m cell.1))
theorem helper_machine_scale_original (base : Nat → Nat) (r : HelperRequest) (laws : TensorLaws base r.weight)
    (a : Ops I F H) (data scales : Nat) (plan : Fin r.G → ReductionPlan r.K) (m : Memory F H) (g : Fin r.G) :
    quantScale a (r.temporary data scales) plan m g=helperMachineScale a r plan m g := by
  have h : (fun k : Fin r.K=>read32 a.toOps m ((r.temporary data scales).inputAddress g k))=
      (fun k : Fin r.K=>helperOriginal a r m g k) := by
    funext k;rw [helper_input_address base r laws data scales g k g.isLt k.isLt];rfl
  unfold quantScale helperMachineScale
  rw [h]
  rfl
theorem helper_machine_value_original (base : Nat → Nat) (r : HelperRequest) (laws : TensorLaws base r.weight)
    (a : Ops I F H) (data scales : Nat) (plan : Fin r.G → ReductionPlan r.K) (m : Memory F H) (cell : QuantCell r.G r.K) :
    quantValue a (r.temporary data scales) plan m cell=helperMachineValue a r plan m cell := by
  unfold quantValue helperMachineValue
  rw [helper_machine_scale_original base r laws,helper_input_address base r laws data scales cell.1 cell.2 cell.1.isLt cell.2.isLt]
  rfl

structure HelperSource (I F H : Type) (r : HelperRequest) (q dest : QuantBuffers) where
  invocation : QuantInvocation r.G r.K
  parsedProduction : String
  sourceIdentity : parsedProduction=Source.productionAST
  lowered : Code I F H Unit
  sourceLowering : lowered=helperEffectCode r q dest invocation
def helperNative (abi : LocalEventABI I F H σ a) (call : HelperSource I F H r q dest) (state : σ) : σ :=
  (call.lowered.native abi state).2
theorem helperNative_refines (abi : LocalEventABI I F H σ a) (call : HelperSource I F H r q dest) (state : σ) :
    abi.observe (helperNative abi call state)=helperEffect a r q dest call.invocation (abi.observe state) := by
  unfold helperNative
  rw [call.sourceLowering]
  have h:=congrArg Prod.snd ((helperEffectCode r q dest call.invocation).native_refines abi state)
  simpa only [Prod.snd,helperEffectCode_eval] using h

structure NativeHelperAllocation (r : HelperRequest) (abi : LocalEventABI I F H σ a) (caller : σ) (live : Nat → Prop) where
  logical : HelperAllocation r (abi.observe caller) live
  state : σ
  observed : abi.observe state=logical.memory
structure NativeHelperResult (σ : Type) where
  values : TensorView
  scales : TensorView
  third : Option Unit
  state : σ
  currentDevice : Nat
def runHelperNative (r : HelperRequest) (abi : LocalEventABI I F H σ a) (callerDevice : Nat)
    (caller : σ) (live : Nat → Prop) (alloc : NativeHelperAllocation r abi caller live)
    (call : HelperSource I F H r alloc.logical.q alloc.logical.dest) : Except InputError (NativeHelperResult σ) := do
  checkHelper (.tensor r.weight.info) (.integer r.ax)
  return ⟨alloc.logical.final,alloc.logical.scales,none,helperNative abi call alloc.state,callerDevice⟩

theorem helper_native_complete (r : HelperRequest) (abi : LocalEventABI I F H σ a) (callerDevice : Nat)
    (caller : σ) (live : Nat → Prop) (input : live r.weight.info.storage) (alloc : NativeHelperAllocation r abi caller live)
    (call : HelperSource I F H r alloc.logical.q alloc.logical.dest) :
    ∃ result,runHelperNative r abi callerDevice caller live alloc call=.ok result ∧
      result.values=alloc.logical.final ∧ result.scales=alloc.logical.scales ∧ result.third=none ∧ result.currentDevice=callerDevice ∧
      (r.needsCopy=Bool.true → ColumnDescriptor r result.values) ∧
      (r.needsCopy=Bool.false → result.values=alloc.logical.temporary) ∧
      FreshDescriptor result.scales (.floating .fp32) [r.G] [1] r.weight.info.metadata.device ∧
      HelperViewBinding r result.values result.scales ∧
      (∀ cell : QuantCell r.G r.K,abi.observe result.state (alloc.logical.dest.dataAddress cell)=.inl (quantValue a alloc.logical.q call.invocation.plan (abi.observe caller) cell)) ∧
      (∀ g : Fin r.G,abi.observe result.state (alloc.logical.q.scaleAddress g)=.inr (.inl (quantScale a alloc.logical.q call.invocation.plan (abi.observe caller) g))) ∧
      (∀ p,live p.1 → abi.observe result.state p=abi.observe caller p) := by
  refine ⟨⟨alloc.logical.final,alloc.logical.scales,none,helperNative abi call alloc.state,callerDevice⟩,
    ?_,rfl,rfl,rfl,rfl,alloc.logical.copyDescriptor,alloc.logical.noCopyIdentity,alloc.logical.scaleDescriptor,alloc.logical.binding,?_,?_,?_⟩
  · simp [runHelperNative,(checkHelper_iff (.tensor r.weight.info) (.integer r.ax)).mpr r.contract,Except.bind,Except.pure,bind,pure]
  · intro cell
    dsimp only
    rw [helperNative_refines,alloc.observed]
    exact helper_values r a call.invocation _ live input alloc.logical cell
  · intro g
    dsimp only
    rw [helperNative_refines,alloc.observed]
    exact helper_scales r a call.invocation _ live input alloc.logical g
  · intro p hp
    dsimp only
    rw [helperNative_refines,alloc.observed]
    exact helper_frame r a call.invocation _ live input alloc.logical p hp

#axiomsClean helper_real_complete
#axiomsClean helperNative_refines
#axiomsClean helper_native_complete
#axiomsClean helper_machine_scale_original
#axiomsClean helper_machine_value_original
#specNonCircular helperMachineValue avoiding [runHelperNative,helperEffect,quantValue,quantScale]

/-- Whole Torch-helper native result, with its own scalar profile and the
actual optional transpose copy; no desired-output field is an assumption. -/
theorem helper_native_if_local (base : Nat → Nat) (r : HelperRequest) (laws : TensorLaws base r.weight)
    (a : Ops (BitVec 32) Word32 Word16) (isa : FixedISAProfile) (scalar : ScalarProfile)
    (numeric : PrimitiveLeaves a isa scalar) (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a)
    (raw : RawByteABI abi) (callerDevice : Nat) (caller : σ) (live : Nat → Prop)
    (input : live r.weight.info.storage) (inputKind : raw.dtype r.weight.info.storage=.fp32)
    (inputWords : TensorWords .fp32 r.weight (abi.observe caller))
    (alloc : NativeHelperAllocation r abi caller live)
    (dataLaws : TensorLaws base alloc.logical.final) (scaleLaws : TensorLaws base alloc.logical.scales)
    (dataKind : raw.dtype alloc.logical.final.info.storage=.int8)
    (scaleKind : raw.dtype alloc.logical.scales.info.storage=.fp32)
    (call : HelperSource (BitVec 32) Word32 Word16 r alloc.logical.q alloc.logical.dest) :
    ∃ result,runHelperNative r abi callerDevice caller live alloc call=.ok result ∧
      result.values=alloc.logical.final ∧ result.scales=alloc.logical.scales ∧ result.third=none ∧ result.currentDevice=callerDevice ∧ InputPointerBinding base r.weight ∧
      InputPointerBinding base result.values ∧ InputPointerBinding base result.scales ∧
      (r.needsCopy=Bool.true → ColumnDescriptor r result.values) ∧
      (r.needsCopy=Bool.false → result.values=alloc.logical.temporary) ∧
      FreshDescriptor result.scales (.floating .fp32) [r.G] [1] r.weight.info.metadata.device ∧
      HelperViewBinding r result.values result.scales ∧
      (∀ cell : QuantCell r.G r.K,raw.bytes result.state (alloc.logical.dest.dataAddress cell)=
        helperMachineValue (fpOps isa scalar) r call.invocation.plan (abi.observe caller) cell) ∧
      (∀ g : Fin r.G,∀ i : Fin 4,raw.bytes result.state (alloc.logical.scales.info.storage,4*g.val+i.val)=
        (helperMachineScale (fpOps isa scalar) r call.invocation.plan (abi.observe caller) g).bits.extractLsb' (8*i.val) 8) ∧
      (∀ storage offset,live storage → raw.bytes result.state (storage,offset)=raw.bytes caller (storage,offset)) := by
  obtain ⟨result,ok,valuesID,scalesID,none,device,copy,nocopy,scaleDesc,readback,values,scales,frame⟩:=
    helper_native_complete r abi callerDevice caller live input alloc call
  refine ⟨result,ok,valuesID,scalesID,none,device,input_pointer_binding base r.weight laws,
    (by rw [valuesID];exact input_pointer_binding base alloc.logical.final dataLaws),
    (by rw [scalesID];exact input_pointer_binding base alloc.logical.scales scaleLaws),copy,nocopy,scaleDesc,readback,?_,?_,?_⟩
  · intro cell
    have eq:=congrArg (fun ops=>quantValue ops alloc.logical.q call.invocation.plan (abi.observe caller) cell) numeric.eq
    have h:=(values cell).trans (congrArg Sum.inl eq)
    dsimp only at h
    change abi.observe result.state (alloc.logical.dest.dataAddress cell)=.inl
      (quantValue (fpOps isa scalar) (r.temporary alloc.logical.temporary.info.storage alloc.logical.scales.info.storage) call.invocation.plan (abi.observe caller) cell) at h
    rw [helper_machine_value_original base r laws] at h
    have kind : raw.dtype alloc.logical.dest.output=.int8:=dataKind
    rw [raw.decode]
    simp only [QuantBuffers.dataAddress,Prod.fst,Prod.snd]
    rw [kind]
    simp only [dtypeBytes,Nat.div_one,Nat.mod_one]
    change cellByte (abi.observe result.state (alloc.logical.dest.dataAddress cell)) 0=_
    rw [h];rfl
  · intro g i
    have eq:=congrArg (fun ops=>quantScale ops alloc.logical.q call.invocation.plan (abi.observe caller) g) numeric.eq
    have h:=(scales g).trans (congrArg (fun x:Word32=>(Sum.inr (Sum.inl x):Cell Word32 Word16)) eq)
    dsimp only at h
    change abi.observe result.state (alloc.logical.q.scaleAddress g)=.inr (.inl
      (quantScale (fpOps isa scalar) (r.temporary alloc.logical.temporary.info.storage alloc.logical.scales.info.storage) call.invocation.plan (abi.observe caller) g)) at h
    rw [helper_machine_scale_original base r laws] at h
    simp only [QuantBuffers.scaleAddress,HelperAllocation.q,HelperRequest.temporary,Nat.zero_add] at h
    rw [raw.decode,scaleKind]
    simp only [dtypeBytes,Nat.mul_add_div (by decide : 0<4),Nat.div_eq_of_lt i.isLt,Nat.add_zero,
      Nat.mul_add_mod,Nat.mul_mod_right,Nat.zero_add,Nat.mod_eq_of_lt i.isLt,h,cellByte]
  · intro storage offset owned
    rw [raw.decode,raw.decode,frame _ owned]

#axiomsClean helper_native_if_local
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.Direct
