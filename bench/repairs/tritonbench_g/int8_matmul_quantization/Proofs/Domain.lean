import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.Native

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization
noncomputable section

structure TensorHandle where
  objectID : Nat
  metadata : Tensor
  isNegative : Bool
  isConjugated : Bool
  deriving DecidableEq
/-- The source does not resolve lazy neg/conj views. Admission therefore
requires the logical view and raw storage to agree; this is not an added
resolve_neg/resolve_conj operation or an all-dtype API claim. -/
structure RawResolved (m : Memory F H) (t : Tensor) where
  logical : Nat → Cell F H
  same : ∀ i,logical i=m (t.storage,t.offset+i)

structure GemmAdmission (s : Shape) (b : Buffers) (owner : Nat)
    (A AS B BS : TensorHandle) (supplied : Option TensorHandle) (returned : TensorHandle)
    (m : Memory Word32 Word16) where
  aMeta : A.metadata=⟨b.a,b.aOffset,[s.M,s.K],[max s.K 1,1],.int8,owner⟩
  bMeta : B.metadata=⟨b.b,b.bOffset,[s.K,s.N],
    if b.bColumnMajor then [1,max s.K 1] else [max s.N 1,1],.int8,owner⟩
  asMeta : AS.metadata=⟨b.aScale,b.aScaleOffset,[s.M],[1],precisionDType b.aScalePrecision,owner⟩
  bsMeta : BS.metadata=⟨b.bScale,b.bScaleOffset,[s.N],[1],precisionDType b.bScalePrecision,owner⟩
  outputMeta : returned.metadata=⟨b.out,b.outOffset,[s.M,s.N],
    returned.metadata.strides,precisionDType b.outPrecision,owner⟩
  /-- Active output addresses use canonical row-major arithmetic. Empty
  supplied views have no active addresses and retain their actual strides. -/
  outputLayout : 0<s.M → 0<s.N → returned.metadata.strides=[s.N,1]
  defaultMeta : supplied=none → returned.metadata=outputTensor s b owner
  outputIdentity : ∀ out,supplied=some out → returned=out
  defaultHalf : supplied=none → b.outPrecision=.half
  separate : b.Separate
  resolvedFlags : ∀ t,t∈[A,AS,B,BS,returned] → t.isNegative=Bool.false ∧ t.isConjugated=Bool.false
  aView : RawResolved m A.metadata
  bView : RawResolved m B.metadata
  asView : RawResolved m AS.metadata
  bsView : RawResolved m BS.metadata
  aTyped : ∀ r k,r<s.M → k<s.K → ∃ byte,m (b.a,b.aOffset+r*s.K+k)=.inl byte
  bTyped : ∀ k n,k<s.K → n<s.N → ∃ byte,m (b.b,b.bIndex s k n)=.inl byte
  asTyped : ∀ r,r<s.M → match b.aScalePrecision with
    | .half => ∃ word,m (b.aScale,b.aScaleOffset+r)=.inr (.inr word)
    | .single => ∃ word,m (b.aScale,b.aScaleOffset+r)=.inr (.inl word)
  bsTyped : ∀ n,n<s.N → match b.bScalePrecision with
    | .half => ∃ word,m (b.bScale,b.bScaleOffset+n)=.inr (.inr word)
    | .single => ∃ word,m (b.bScale,b.bScaleOffset+n)=.inr (.inl word)

structure RowAdmission (G K : Nat) (q : QuantBuffers) (owner : Nat)
    (input output scale : TensorHandle) (m : Memory Word32 Word16) where
  inputMeta : input.metadata=⟨q.input,q.inputOffset,[G,K],[q.inputGroupStride,q.inputElementStride],.fp32,owner⟩
  outputMeta : output.metadata=⟨q.output,q.outputOffset,[G,K],[max K 1,1],.int8,owner⟩
  scaleMeta : scale.metadata=⟨q.scales,q.scaleOffset,[G],[1],.fp16,owner⟩
  layout : q.transposeOutput=Bool.false
  precision : q.scalePrecision=.half
  initialZero : q.initialMaximumZero=Bool.true
  inputView : RawResolved m input.metadata
  inputTyped : ∀ g k,g<G → k<K → ∃ word,m (q.inputAddress g k)=.inr (.inl word)
  separate : q.Separate
  domain : G=0 ∨ 0<K
  resolvedFlags : ∀ t,t∈[input,output,scale] → t.isNegative=Bool.false ∧ t.isConjugated=Bool.false

def cellByte (cell : Cell Word32 Word16) (i : Nat) : Byte :=
  match cell with
  | .inl byte => byte
  | .inr (.inl word) => word.bits.extractLsb' (8*i) 8
  | .inr (.inr word) => word.bits.extractLsb' (8*i) 8
def dtypeBytes : DType → Nat | .int8 => 1 | .fp16 => 2 | .fp32 => 4
/-- A per-element byte representation contract; these are projections of
typed observed memory, not a separately implemented byte-memory interpreter. -/
structure RawByteABI (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a) where
  dtype : Nat → DType
  bytes : σ → Address → Byte
  decode : ∀ state storage offset,bytes state (storage,offset)=
    cellByte (abi.observe state (storage,offset/dtypeBytes (dtype storage))) (offset%dtypeBytes (dtype storage))
theorem same_raw_byte (abi : LocalEventABI (BitVec 32) Word32 Word16 σ a) (raw : RawByteABI abi)
    (state : σ) (storage offset : Nat) (kind : raw.dtype storage=.int8)
    (typed : ∃ value,abi.observe state (storage,offset)=.inl value) :
    read8 (abi.observe state) (storage,offset)=raw.bytes state (storage,offset) := by
  obtain ⟨value,typed⟩ := typed
  rw [raw.decode,kind]
  simp [dtypeBytes,cellByte,read8,Int8DequantMatmul.read8,typed]

structure AllocationABI (caller prepared : Memory Word32 Word16) (newStorage : Nat → Prop) : Prop where
  frame : ∀ p,¬newStorage p.1 → prepared p=caller p

/-- The allocated or supplied handle and its exact metadata are returned
directly. The compiled memory/device execution is independent of inert
empty-view strides; its canonical allocation descriptor is replaced by
the actual host handle's descriptor, without mutating that handle. -/
def publicGemm (returned : TensorHandle) (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi)
    (owner callerDevice : Nat) (calls : List (BoundCall I F H s b))
    (chosen : BoundCall I F H s b) (state : σ) :=
  (returned,{ compiledWrapper abi hook owner callerDevice calls chosen state with tensor := returned.metadata })
theorem supplied_output_identity (admit : GemmAdmission s b owner A AS B BS (some out) returned m)
    (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi) (callerDevice : Nat)
    (calls : List (BoundCall I F H s b)) (chosen : BoundCall I F H s b) (state : σ) :
    (publicGemm returned abi hook owner callerDevice calls chosen state).1=out := admit.outputIdentity out rfl

/-- This metadata result has no MatrixCell premise and covers empty outputs. -/
theorem public_output_metadata (returned : TensorHandle)
    (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi) (owner callerDevice : Nat)
    (calls : List (BoundCall I F H s b)) (chosen : BoundCall I F H s b) (state : σ) :
    (publicGemm returned abi hook owner callerDevice calls chosen state).2.tensor=returned.metadata := rfl

theorem supplied_output_metadata (admit : GemmAdmission s b owner A AS B BS (some out) returned m)
    (abi : LocalEventABI I F H σ a) (hook : ZeroHookABI abi) (callerDevice : Nat)
    (calls : List (BoundCall I F H s b)) (chosen : BoundCall I F H s b) (state : σ) :
    (publicGemm returned abi hook owner callerDevice calls chosen state).2.tensor=out.metadata := by
  rw [public_output_metadata,admit.outputIdentity out rfl]

#axiomsClean same_raw_byte
#axiomsClean supplied_output_identity
#axiomsClean public_output_metadata
#axiomsClean supplied_output_metadata
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization
