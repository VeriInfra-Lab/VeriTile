import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.DirectPointers
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.DirectNative
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.DirectWords

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.Direct
noncomputable section
set_option maxHeartbeats 1500000

structure NativeRowAllocation (r : RowRequest) (abi : LocalEventABI I F H σ a) (caller : σ) (live : Nat → Prop) where
  logical : RowAllocation r (abi.observe caller) live
  state : σ
  observed : abi.observe state=logical.memory
structure NativeRowResult (σ : Type) where
  data : TensorView
  scales : TensorView
  state : σ
  currentDevice : Nat
def runRowNative (r : RowRequest) (abi : LocalEventABI I F H σ a) (callerDevice : Nat)
    (caller : σ) (live : Nat → Prop) (alloc : NativeRowAllocation r abi caller live)
    (trials : List (RowSource I F H r.G r.K (r.buffers alloc.logical.data.info.storage alloc.logical.scales.info.storage)))
    (chosen : RowSource I F H r.G r.K (r.buffers alloc.logical.data.info.storage alloc.logical.scales.info.storage)) :
    Except InputError (NativeRowResult σ) := do
  checkRow (.tensor r.input.info)
  return ⟨alloc.logical.data,alloc.logical.scales,rowNative r _ _ abi trials chosen alloc.state,callerDevice⟩

theorem row_native_complete (r : RowRequest) (abi : LocalEventABI I F H σ a) (callerDevice : Nat)
    (caller : σ) (live : Nat → Prop) (input : live r.input.info.storage) (alloc : NativeRowAllocation r abi caller live)
    (trials : List (RowSource I F H r.G r.K (r.buffers alloc.logical.data.info.storage alloc.logical.scales.info.storage)))
    (chosen : RowSource I F H r.G r.K (r.buffers alloc.logical.data.info.storage alloc.logical.scales.info.storage)) :
    ∃ result,runRowNative r abi callerDevice caller live alloc trials chosen=.ok result ∧
      result.data=alloc.logical.data ∧ result.scales=alloc.logical.scales ∧
      FreshDescriptor result.data (.other "int8") [r.G,r.K] [max r.K 1,1] r.input.info.metadata.device ∧
      FreshDescriptor result.scales (.other "float16") [r.G] [1] r.input.info.metadata.device ∧
      RowViewBinding r result.data result.scales ∧ result.currentDevice=callerDevice ∧
      (∀ idx : QuantIndex r.G r.K,abi.observe result.state ((r.buffers result.data.info.storage result.scales.info.storage).writeAddress idx)=
        quantExpected a (r.buffers result.data.info.storage result.scales.info.storage) chosen.invocation.plan (abi.observe caller) idx) ∧
      (∀ p,live p.1 → abi.observe result.state p=abi.observe caller p) := by
  let q:=r.buffers alloc.logical.data.info.storage alloc.logical.scales.info.storage
  have sep : q.Separate:=alloc.logical.separate r _ live input
  have same : QuantInputsAgree q alloc.logical.memory (abi.observe caller) := by
    intro p hp
    apply alloc.logical.frame
    · simpa only [hp] using sep.1
    · simpa only [hp] using sep.2.1
  refine ⟨⟨alloc.logical.data,alloc.logical.scales,rowNative r _ _ abi trials chosen alloc.state,callerDevice⟩,
    ?_,rfl,rfl,alloc.logical.dataDescriptor,alloc.logical.scaleDescriptor,alloc.logical.binding,rfl,?_,?_⟩
  · simp [runRowNative,(checkRow_iff (.tensor r.input.info)).mpr r.contract,Except.bind,Except.pure,bind,pure]
  · intro idx
    dsimp only
    rw [rowNative_refines,alloc.observed]
    change quantWrapper a q _ _ alloc.logical.memory (q.writeAddress idx)=quantExpected a q _ _ idx
    rw [quantWrapper_at a q sep]
    cases idx with
    | inl cell=>exact congrArg Sum.inl (quantValue_current a q _ _ _ same cell)
    | inr g=>simp only [quantExpected,quantScale_current a q _ _ _ same]
  · intro p hp
    dsimp only
    rw [rowNative_refines,alloc.observed]
    have hd : p.1≠alloc.logical.data.info.storage:=by intro h;rw [h] at hp;exact alloc.logical.dataFresh hp
    have hs : p.1≠alloc.logical.scales.info.storage:=by intro h;rw [h] at hp;exact alloc.logical.scaleFresh hp
    change quantWrapper a q _ _ alloc.logical.memory p=abi.observe caller p
    rw [quantWrapper_frame a q sep]
    · exact alloc.logical.frame p hd hs
    · intro idx h
      have first:=congrArg Prod.fst h
      cases idx with
      | inl x=>exact hd first
      | inr g=>exact hs first

/-- Input representation is a local Tensor/storage boundary. All payloads,
including nonfinite floating words, remain admitted. -/
def TensorWords (dtype : DType) (t : TensorView) (m : Memory Word32 Word16) : Prop :=
  ∀ i,i<t.info.metadata.numel → match dtype with
    | .int8=>∃ value,m (logicalAddress t i)=.inl value
    | .fp16=>∃ value,m (logicalAddress t i)=.inr (.inr value)
    | .fp32=>∃ value,m (logicalAddress t i)=.inr (.inl value)
structure GemmInputABI (r : GemmRequest) (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a)
    (raw : RawByteABI abi) (state : σ) : Prop where
  aKind : raw.dtype r.a.info.storage=.int8
  bKind : raw.dtype r.b.info.storage=.int8
  asKind : raw.dtype r.aScale.info.storage=precisionDType (scalePrecision (some r.aScale.info) Bool.false)
  bsKind : raw.dtype r.bScale.info.storage=.fp32
  aTyped : TensorWords .int8 r.a (abi.observe state)
  bTyped : TensorWords .int8 r.b (abi.observe state)
  asTyped : TensorWords (precisionDType (scalePrecision (some r.aScale.info) Bool.false)) r.aScale (abi.observe state)
  bsTyped : TensorWords .fp32 r.bScale (abi.observe state)
structure RowInputABI (r : RowRequest) (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a)
    (raw : RawByteABI abi) (state : σ) : Prop where
  kind : raw.dtype r.input.info.storage=.fp32
  typed : TensorWords .fp32 r.input (abi.observe state)

theorem dtypeBytes_positive (dtype : DType) : 0<dtypeBytes dtype := by cases dtype <;> decide

#axiomsClean row_native_complete
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.Direct
