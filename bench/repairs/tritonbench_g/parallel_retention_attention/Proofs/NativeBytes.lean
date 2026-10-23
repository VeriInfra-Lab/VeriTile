import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.HistoricalFPEquiv
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.Checked
namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.NativeBytes
open Classical
noncomputable section
abbrev Word := FP32Word
abbrev Byte := BitVec 8
/-- Byte-addressed backing memory; live addresses are bounded by PointerBinding
and the checked signed64 pointer room, proved below. -/
abbrev Bytes := Nat → Byte

def encode (x : Word) (i : Fin 4) : Byte := x.bits.extractLsb' (8*i.val) 8
def decode (bytes : Fin 4 → Byte) : Word := ⟨bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0⟩

theorem decode_encode (x : Word) : decode (encode x)=x := by
  change VeriTile.Triton.FP.Value.mk _=VeriTile.Triton.FP.Value.mk x.bits
  congr 1
  change (x.bits.extractLsb' 24 8 ++ x.bits.extractLsb' 16 8 ++ x.bits.extractLsb' 8 8 ++ x.bits.extractLsb' 0 8)=x.bits
  rw [BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 24=16+8),
    BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 16=8+8),
    BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 8=0+8)]
  exact BitVec.extractLsb'_eq_self

theorem encode_decode (bytes : Fin 4 → Byte) : encode (decode bytes)=bytes := by
  funext i
  fin_cases i
  · change ((bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0) : BitVec 32).extractLsb' 0 8=bytes 0
    exact BitVec.extractLsb'_append_eq_right
  · change ((bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0) : BitVec 32).extractLsb' 8 8=bytes 1
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8≤8)]
    exact BitVec.extractLsb'_append_eq_right
  · change ((bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0) : BitVec 32).extractLsb' 16 8=bytes 2
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8≤16),
      BitVec.extractLsb'_append_eq_of_le (by decide : 8≤8)]
    exact BitVec.extractLsb'_append_eq_right
  · change ((bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0) : BitVec 32).extractLsb' 24 8=bytes 3
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8≤24),
      BitVec.extractLsb'_append_eq_of_le (by decide : 8≤16)]
    exact BitVec.extractLsb'_append_eq_left

def load (m : Bytes) (wordAddress : Nat) : Word := decode (fun i=>m (4*wordAddress+i.val))

theorem original_bytes (m : Bytes) (a : Nat) (i : Fin 4) : encode (load m a) i=m (4*a+i.val) :=
  congrFun (encode_decode (fun i=>m (4*a+i.val))) i

/-- One little-endian aligned FP32 store. Exactly four backing bytes change. -/
def store (m : Bytes) (wordAddress : Nat) (v : Word) : Bytes :=
  fun a=>if a/4=wordAddress then encode v ⟨a%4,Nat.mod_lt _ (by decide)⟩ else m a

theorem load_store (m : Bytes) (a b : Nat) (v : Word) :
    load (store m a v) b=if b=a then v else load m b := by
  have div (i : Fin 4) : (4*b+i.val)/4=b := by omega
  have rem (i : Fin 4) : (4*b+i.val)%4=i.val := by omega
  unfold load store
  simp only [div]
  split_ifs with h
  · have eq : (fun i : Fin 4=>encode v ⟨(4*b+i.val)%4,Nat.mod_lt _ (by decide)⟩)=encode v := by
      funext i;congr 1;exact Fin.ext (rem i)
    rw [eq,decode_encode]
  · rfl

theorem store_frame (m : Bytes) (a byte : Nat) (v : Word) (outside : byte/4≠a) :
    store m a v byte=m byte := by simp [store,outside]


def runtime : LocalRuntime Word Bytes where
  observe := load
  write := store
  write_at := fun m a v b => load_store m a b v

theorem byte_eq (m : Bytes) (byte : Nat) :
    m byte=encode (load m (byte/4)) ⟨byte%4,Nat.mod_lt _ (by decide)⟩ := by
  have h := original_bytes m (byte/4) ⟨byte%4,Nat.mod_lt _ (by decide)⟩
  have arith : 4*(byte/4)+byte%4=byte := by omega
  simpa only [arith] using h.symm

/-- The four ordinary Tensor stride observations used only for the original
Torch contiguous copies. Kernel addressing remains canonical. -/
structure View where
  batchStride : Nat
  headStride : Nat
  timeStride : Nat
  featureStride : Nat

def View.byteAddress (v : View) (base H bh t e : Nat) : Nat :=
  base+4*((bh/H)*v.batchStride+(bh%H)*v.headStride+t*v.timeStride+e*v.featureStride)

def loadAt (m : Bytes) (byte : Nat) : Word := decode (fun j=>m (byte+j.val))

/-- Exact input words come from the original Tensor data pointers and their
ordinary strides. These premises concern entry-state reads only, including
the actual incoming cotangent at the later backward call. -/
structure OriginalInputs (s : Shape) (o : Checked.Inputs) (forwardEntry backwardEntry : Bytes)
    (x : LogicalInputs Word s) where
  qView : View
  kView : View
  vView : View
  gradient : Checked.Tensor
  gView : View
  gradientValid : gradient.Valid
  gradientShape : gradient.metadata.shape=[s.B,s.H,s.T,s.DV]
  gradientDevice : gradient.metadata.device=o.query.metadata.device
  query : ∀ bh t e,x.q bh t e=loadAt forwardEntry
    (qView.byteAddress o.query.metadata.dataPtr.toNat s.H bh t e)
  key : ∀ bh t e,x.k bh t e=loadAt forwardEntry
    (kView.byteAddress o.key.metadata.dataPtr.toNat s.H bh t e)
  value : ∀ bh t e,x.v bh t e=loadAt forwardEntry
    (vView.byteAddress o.value.metadata.dataPtr.toNat s.H bh t e)
  cotangent : ∀ bh t e,x.g bh t e=loadAt backwardEntry
    (gView.byteAddress gradient.metadata.dataPtr.toNat s.H bh t e)
  sourceBounds : (∀ bh : Fin (s.B*s.H),∀ t : Fin s.T,∀ e : Fin s.DK,
    qView.byteAddress o.query.metadata.dataPtr.toNat s.H bh t e+4≤2^64 ∧
    kView.byteAddress o.key.metadata.dataPtr.toNat s.H bh t e+4≤2^64) ∧
    (∀ bh : Fin (s.B*s.H),∀ t : Fin s.T,∀ e : Fin s.DV,
    vView.byteAddress o.value.metadata.dataPtr.toNat s.H bh t e+4≤2^64 ∧
    gView.byteAddress gradient.metadata.dataPtr.toNat s.H bh t e+4≤2^64)

/-- Input-only contract of the original contiguous materializations. The
footprint contains their fresh copied storage; no kernel output or future
memory state occurs. Already-contiguous inputs use an empty footprint. -/
structure Copies (entry ready : Bytes) where
  buffers : List (Nat×Nat)
  frame : ∀ byte,(∀ buffer∈buffers,¬(buffer.1≤byte ∧ byte<buffer.1+buffer.2)) → ready byte=entry byte

def Copies.footprint {entry ready : Bytes} (copy : Copies entry ready) : Set Nat :=
  {byte | ∃ buffer∈copy.buffers,buffer.1≤byte ∧ byte<buffer.1+buffer.2}

theorem Copies.outside_frame {entry ready : Bytes} (copy : Copies entry ready) (byte : Nat)
    (outside : byte∉copy.footprint) : ready byte=entry byte := by
  apply copy.frame byte
  intro buffer hb h
  exact outside ⟨buffer,hb,h⟩

/-- Metadata of the actual returned allocation, with no value or execution
premise. Base includes any storage offset and denotes four-byte words. -/
structure Returned (handle : TensorHandle) where
  metadata : VeriTile.Torch.TensorMetadata
  shape : metadata.shape=handle.shape
  dtype : metadata.dtype=.floating .fp32
  layout : metadata.layout=.strided
  device : metadata.device=⟨"cuda",handle.device⟩
  contiguous : metadata.isContiguous=true
  logical : metadata.isNeg=false
  pointer : handle.shape.prod>0 → metadata.dataPtr.toNat=4*handle.base
  room : handle.shape.prod>0 → 4*(handle.base+handle.shape.prod)<2^64

theorem returned_pointer (s : Shape) (kind : OutputKind) (base device : Nat)
    (t : Returned (outputHandle s kind base device)) (i : LogicalIndex s kind) :
    t.metadata.dataPtr.toNat=4*base := by
  apply t.pointer
  simp only [outputHandle,List.prod_cons,List.prod_nil,Nat.mul_one]
  have hb := i.batch.isLt;have hh := i.head.isLt
  have ht := i.time.isLt;have he := i.feature.isLt
  exact Nat.mul_pos (by omega) (Nat.mul_pos (by omega) (Nat.mul_pos (by omega) (by omega)))

theorem returned_byte_bound (s : Shape) (kind : OutputKind) (base device : Nat)
    (t : Returned (outputHandle s kind base device)) (i : LogicalIndex s kind) (j : Fin 4) :
    t.metadata.dataPtr.toNat+4*dense s.T (kind.width s) i.bh i.time i.feature+j.val<2^64 := by
  have bound := dense_bound s.T (kind.width s) (s.B*s.H) i.bh i.time i.feature
    i.bh_bound i.time.isLt i.feature.isLt
  have count : (outputHandle s kind base device).shape.prod=s.B*s.H*s.T*kind.width s := by
    simp [outputHandle,Nat.mul_assoc]
  have room := t.room (by rw [count];omega)
  rw [count] at room
  dsimp only [outputHandle] at room
  rw [returned_pointer s kind base device t i]
  have hj := j.isLt
  omega

/-- Byte lookup at the actual returned data pointer, including the output
coordinate and the little-endian component. -/
theorem returned_byte (s : Shape) (kind : OutputKind) (base device : Nat)
    (t : Returned (outputHandle s kind base device)) (i : LogicalIndex s kind)
    (m : Bytes) (word : Word)
    (read : readReturned s kind (outputHandle s kind base device) (load m) i=word)
    (j : Fin 4) :
    m (t.metadata.dataPtr.toNat+4*dense s.T (kind.width s) i.bh i.time i.feature+j.val)=encode word j := by
  rw [returned_pointer s kind base device t i]
  have h := original_bytes m (base+dense s.T (kind.width s) i.bh i.time i.feature) j
  change load m (base+dense s.T (kind.width s) i.bh i.time i.feature)=word at read
  rw [read] at h
  simpa only [Nat.mul_add,Nat.add_assoc] using h.symm

theorem frame_of_words (before after : Bytes) (byte : Nat)
    (h : load after (byte/4)=load before (byte/4)) : after byte=before byte := by
  rw [byte_eq after byte,byte_eq before byte,h]

#axiomsClean decode_encode
#axiomsClean encode_decode
#axiomsClean load_store
#axiomsClean returned_byte
#axiomsClean returned_byte_bound
#axiomsClean frame_of_words
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.NativeBytes
