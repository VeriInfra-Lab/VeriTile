import bench.tritonbench_g.attention_llama.Proofs.CheckedCorrect
import bench.tritonbench_g.attention_llama.Proofs.SourceFPEquiv

namespace VeriTile.Bench.Repairs.AttentionLlama.Checked
open VeriTile Triton
open LoopInvariant HostGrid HostMemory Correctness FPEquivalence
set_option maxHeartbeats 1500000

abbrev Byte := BitVec 8
abbrev HalfWord := BitVec 16
abbrev ByteMemory := Fin (2^64)→Byte

def splitHalf (w : HalfWord) (j : Fin 2) : Byte := BitVec.ofNat 8 (w.toNat/256^j.val)
def joinHalf (bytes : Fin 2→Byte) : HalfWord :=
  BitVec.ofNat 16 ((bytes 0).toNat+256*(bytes 1).toNat)

theorem join_split (w : HalfWord) : joinHalf (splitHalf w)=w := by
  apply BitVec.eq_of_toNat_eq
  have hw:=w.isLt
  simp only [joinHalf,splitHalf,BitVec.toNat_ofNat,Fin.val_zero,Fin.val_one,pow_zero,pow_one,Nat.div_one]
  norm_num at hw ⊢
  omega

/-- Ordinary FP16 word codec. Its only round-trip obligation is at a single
FP16 conversion, never an attention output or numerical-to-Real equality. -/
structure HalfCodec {α : Type} (A : NativeGraph.Ops α) where
  decode : HalfWord→α
  encode : α→HalfWord
  halfRoundtrip : ∀ x, decode (encode (A.half x))=A.half x

/-- Ordinary scalar marshalling retains the actual Python scalar and the
word passed to the kernel ABI. `decode` interprets this native parameter word. -/
inductive RawScalar where
  | integer (value : Int)
  | float64 (bits : BitVec 64)
  deriving DecidableEq

structure ScalarWord (α : Type) where
  argument : RawScalar
  word : BitVec 32
  marshal : RawScalar→BitVec 32
  project : RawScalar→ℝ
  decode : BitVec 32→α
  marshalled : word=marshal argument

def ScalarWord.value {α : Type} (s : ScalarWord α) := s.decode s.word

def extendMemory (memory : ByteMemory) (n : Nat) : Byte :=
  if h : n<2^64 then memory ⟨n,h⟩ else 0

def readHalfAt (memory : ByteMemory) (base : Nat) : HalfWord :=
  joinHalf (fun j=>extendMemory memory (base+j.val))

/-- Q/K/V are read from actual physical bytes at the caller's actual pointers
and strides. These are coordinate observations only. Source scale comes from
actual scalar marshalling, not an independent α parameter. -/
def observedInputs {α : Type} (B : Nat) (c : Context) (o : OutputLayout B c)
    (A : NativeGraph.Ops α) (codec : HalfCodec A) (p : Pointers B c o) (memory : ByteMemory) :
    FPEquivalence.TensorInputs α B c.NH c.H c.D where
  query b h n d:=codec.decode (readHalfAt memory
    (p.base .query+2*(queryStrides c).at b.val h.val n.val d.val))
  halfKey b h n d:=codec.decode (readHalfAt memory
    (p.base .key+2*(keyStrides c).at b.val h.val n d.val))
  halfValue b h n d:=codec.decode (readHalfAt memory
    (p.base .value+2*(valueStrides c).at b.val h.val n d.val))
  byteKey b h n d:=UInt8.ofNat (extendMemory memory
    (p.base .key+(keyStrides c).at b.val h.val n d.val)).toNat
  byteValue b h n d:=UInt8.ofNat (extendMemory memory
    (p.base .value+(valueStrides c).at b.val h.val n d.val)).toNat

/-- Actual Tensor object observations, including storage identity, pointer,
dtype, device, logical shape and independently allocated strides. -/
structure TensorDescriptor where
  (object storage : Nat)
  (dataPointer device allocationBase allocationBytes : Nat)
  dtype : DType
  (shape strides : List Nat)

structure OutputBinding (B : Nat) (c : Context) (o : OutputLayout B c)
    (p : Pointers B c o) (q : TensorMetadata) (allocated : TensorDescriptor) : Prop where
  pointer : allocated.dataPointer=p.base .output
  device : allocated.device=q.device
  dtype : allocated.dtype=DType.half
  shape : allocated.shape=[B,c.NH,c.H,c.D]
  strides : allocated.strides=[o.soz,o.soh,o.som,o.son]
  lower : allocated.allocationBase≤allocated.dataPointer
  backed : allocated.dataPointer+2*(outputStrides o).span B c.NH c.H c.D≤
    allocated.allocationBase+allocated.allocationBytes

abbrev ByteIndex (B : Nat) (c : Context) := Index B c.NH c.H c.D × Fin 2

def outputByte (B : Nat) (c : Context) (o : OutputLayout B c) (i : ByteIndex B c) : Nat :=
  2*outputAddress c o.base o.soz o.soh o.som o.son i.1+i.2.val

theorem outputByte_injective (B : Nat) (c : Context) (o : OutputLayout B c) :
    Function.Injective (outputByte B c o) := by
  intro x y he
  have hx:=x.2.isLt
  have hy:=y.2.isLt
  have ha : outputAddress c o.base o.soz o.soh o.som o.son x.1=
      outputAddress c o.base o.soz o.soh o.som o.son y.1 := by unfold outputByte at he; omega
  have hi:=o.injective ha
  apply Prod.ext hi
  apply Fin.ext
  unfold outputByte at he
  rw [ha] at he
  omega

noncomputable def byteSchedule (B : Nat) (c : Context) : List (ByteIndex B c) :=
  (schedule B c.NH c.H c.D).product [0,1]

theorem byteSchedule_covers (B : Nat) (c : Context) (i : ByteIndex B c) :
    i∈byteSchedule B c := by
  apply List.mem_product.mpr
  refine ⟨schedule_covers i.1,?_⟩
  have h:=i.2.isLt
  have hi : i.2=0 ∨ i.2=1 := by omega
  rcases hi with h|h <;> simp [h]

theorem outputByte_bounded (B : Nat) (c : Context) (o : OutputLayout B c)
    (fp8 : Bool) (p : Pointers B c o) (hp : pointerChecks B c o fp8 p=Bool.true)
    (i : ByteIndex B c) : outputByte B c o i<2^64 := by
  have h:=returned_address B c o fp8 p hp i.1
  have hj:=i.2.isLt
  unfold outputByte
  omega

def descriptorOffset (allocated : TensorDescriptor) (z h n d : Nat) : Nat :=
  allocated.dataPointer+2*(z*allocated.strides[0]?.getD 0+h*allocated.strides[1]?.getD 0+
    n*allocated.strides[2]?.getD 0+d*allocated.strides[3]?.getD 0)

theorem outputByte_descriptor (B : Nat) (c : Context) (o : OutputLayout B c)
    (fp8 : Bool) (p : Pointers B c o) (hp : pointerChecks B c o fp8 p=Bool.true)
    (q : TensorMetadata) (allocated : TensorDescriptor) (binding : OutputBinding B c o p q allocated)
    (i : ByteIndex B c) : outputByte B c o i=
      descriptorOffset allocated i.1.batch.val i.1.head.val i.1.query.val i.1.channel.val+i.2.val := by
  have h:=pointer_formed B c o fp8 p hp .output 0 (by unfold envelope; omega)
  have hdiv:=Nat.mod_add_div (p.base .output) 2
  have he : 2*(p.base .output/2)=p.base .output := by simp only [width] at h; omega
  simp [outputByte,outputAddress_eq,descriptorOffset,binding.pointer,binding.strides,p.outputBinding,Nat.mul_add,he]

structure ReturnedBytes where
  tensor : TensorDescriptor
  memory : ByteMemory

/-- Source stores split every FP16 word into its two physical bytes in the
actual covered grid/lane schedule. The caller returns the allocated object. -/
noncomputable def finishBytes (B : Nat) (c : Context) (o : OutputLayout B c)
    (allocated : TensorDescriptor) (word : Index B c.NH c.H c.D→HalfWord)
    (memory : ByteMemory) : ReturnedBytes :=
  ⟨allocated,fun address=>storeAll (outputByte B c o) (fun i=>splitHalf (word i.1) i.2)
    (extendMemory memory) (byteSchedule B c) address.val⟩

theorem finish_byte (B : Nat) (c : Context) (o : OutputLayout B c)
    (allocated : TensorDescriptor) (word : Index B c.NH c.H c.D→HalfWord) (memory : ByteMemory)
    (fp8 : Bool) (p : Pointers B c o) (hp : pointerChecks B c o fp8 p=Bool.true)
    (i : ByteIndex B c) :
    (finishBytes B c o allocated word memory).memory
      ⟨outputByte B c o i,outputByte_bounded B c o fp8 p hp i⟩=splitHalf (word i.1) i.2 := by
  exact (HostMemory.readback (outputByte B c o) _ (outputByte_injective B c o) (byteSchedule B c)
    (extendMemory memory) i).trans (if_pos (byteSchedule_covers B c i))

theorem finish_frame (B : Nat) (c : Context) (o : OutputLayout B c)
    (allocated : TensorDescriptor) (word : Index B c.NH c.H c.D→HalfWord) (memory : ByteMemory)
    (address : Fin (2^64)) (h : ∀ i, address.val≠outputByte B c o i) :
    (finishBytes B c o allocated word memory).memory address=memory address := by
  change storeAll _ _ _ _ address.val=_
  rw [HostMemory.frame _ _ _ _ address.val (fun i _=>h i)]
  exact dif_pos address.isLt

#axiomsClean join_split
#axiomsClean outputByte_injective
#axiomsClean byteSchedule_covers
#axiomsClean outputByte_bounded
#axiomsClean outputByte_descriptor
#axiomsClean finish_byte
#axiomsClean finish_frame

/-- The actual caller objects bind descriptor pointers, shape and strides to
exactly the values passed to the source. No result or content condition occurs. -/
structure InputObjects (B : Nat) (c : Context) (o : OutputLayout B c)
    (fp8 : Bool) (p : Pointers B c o) where
  (query key value : TensorDescriptor)
  queryPointer : query.dataPointer=p.base .query
  keyPointer : key.dataPointer=p.base .key
  valuePointer : value.dataPointer=p.base .value
  queryShape : query.shape=[B,c.NH,c.H,c.D]
  keyShape : key.shape=[B,c.NH,c.N,c.D]
  valueShape : value.shape=[B,c.NH,c.N,c.D]
  queryStrides : query.strides=[c.sqz,c.sqh,c.sqm,c.sqk]
  keyStrides : key.strides=[c.skz,c.skh,c.skn,c.skk]
  valueStrides : value.strides=[c.svz,c.svh,c.svk,c.svn]
  queryType : query.dtype=.half
  keyType : key.dtype=if fp8 then .int8 else .half
  valueType : value.dtype=if fp8 then .int8 else .half

def ReturnedBytes.read {B NH H D : Nat} (result : ReturnedBytes) (i : Index B NH H D) : HalfWord :=
  readHalfAt result.memory (descriptorOffset result.tensor i.batch.val i.head.val i.query.val i.channel.val)

theorem finish_read (B : Nat) (c : Context) (o : OutputLayout B c)
    (allocated : TensorDescriptor) (word : Index B c.NH c.H c.D→HalfWord) (memory : ByteMemory)
    (fp8 : Bool) (p : Pointers B c o) (hp : pointerChecks B c o fp8 p=Bool.true)
    (q : TensorMetadata) (binding : OutputBinding B c o p q allocated) (i : Index B c.NH c.H c.D) :
    (finishBytes B c o allocated word memory).read i=word i := by
  unfold ReturnedBytes.read readHalfAt
  change joinHalf (fun j=>extendMemory (finishBytes B c o allocated word memory).memory
    (descriptorOffset allocated i.batch.val i.head.val i.query.val i.channel.val+j.val))=_
  have he : (fun j : Fin 2=>extendMemory (finishBytes B c o allocated word memory).memory
      (descriptorOffset allocated i.batch.val i.head.val i.query.val i.channel.val+j.val))=splitHalf (word i) := by
    funext j
    rw [←outputByte_descriptor B c o fp8 p hp q allocated binding (i,j)]
    rw [extendMemory,dif_pos (outputByte_bounded B c o fp8 p hp (i,j))]
    exact finish_byte B c o allocated word memory fp8 p hp (i,j)
  rw [he,join_split]

#axiomsClean finish_read

def descriptorElement (t : TensorDescriptor) (z h n d : Nat) : Nat :=
  z*t.strides[0]?.getD 0+h*t.strides[1]?.getD 0+n*t.strides[2]?.getD 0+d*t.strides[3]?.getD 0

def actualInputs {α : Type} (B : Nat) (c : Context) (o : OutputLayout B c)
    (A : NativeGraph.Ops α) (codec : HalfCodec A) (fp8 : Bool) (p : Pointers B c o)
    (objects : InputObjects B c o fp8 p) (memory : ByteMemory) :
    FPEquivalence.TensorInputs α B c.NH c.H c.D where
  query b h n d:=codec.decode (readHalfAt memory
    (descriptorOffset objects.query b.val h.val n.val d.val))
  halfKey b h n d:=codec.decode (readHalfAt memory
    (descriptorOffset objects.key b.val h.val n d.val))
  halfValue b h n d:=codec.decode (readHalfAt memory
    (descriptorOffset objects.value b.val h.val n d.val))
  byteKey b h n d:=UInt8.ofNat (extendMemory memory
    (objects.key.dataPointer+descriptorElement objects.key b.val h.val n d.val)).toNat
  byteValue b h n d:=UInt8.ofNat (extendMemory memory
    (objects.value.dataPointer+descriptorElement objects.value b.val h.val n d.val)).toNat

theorem actual_inputs_bind {α : Type} (B : Nat) (c : Context) (o : OutputLayout B c)
    (A : NativeGraph.Ops α) (codec : HalfCodec A) (fp8 : Bool) (p : Pointers B c o)
    (objects : InputObjects B c o fp8 p) (memory : ByteMemory) :
    actualInputs B c o A codec fp8 p objects memory=observedInputs B c o A codec p memory := by
  simp only [actualInputs,observedInputs,descriptorOffset,descriptorElement,
    objects.queryPointer,objects.keyPointer,objects.valuePointer,
    objects.queryStrides,objects.keyStrides,objects.valueStrides,
    queryStrides,keyStrides,valueStrides,Strides.at]
  rfl

def FreshInputs {B : Nat} {c : Context} {o : OutputLayout B c} {fp8 : Bool} {p : Pointers B c o}
    (objects : InputObjects B c o fp8 p) (allocated : TensorDescriptor) : Prop :=
  ∀ t∈[objects.query,objects.key,objects.value],
    t.allocationBase+t.allocationBytes≤allocated.allocationBase ∨
    allocated.allocationBase+allocated.allocationBytes≤t.allocationBase

theorem output_within_allocation (B : Nat) (c : Context) (o : OutputLayout B c)
    (fp8 : Bool) (p : Pointers B c o) (hp : pointerChecks B c o fp8 p=Bool.true)
    (q : TensorMetadata) (allocated : TensorDescriptor) (binding : OutputBinding B c o p q allocated)
    (i : ByteIndex B c) :
    allocated.allocationBase≤outputByte B c o i ∧
    outputByte B c o i<allocated.allocationBase+allocated.allocationBytes := by
  have ha:=active_bound (outputStrides o) B c.NH c.H c.D i.1.batch.val i.1.head.val
    i.1.query.val i.1.channel.val i.1.batch.isLt i.1.head.isLt i.1.query.isLt i.1.channel.isLt
  have hb:=binding.backed
  have hl:=binding.lower
  have hj:=i.2.isLt
  rw [outputByte_descriptor B c o fp8 p hp q allocated binding i]
  simp only [descriptorOffset,binding.strides,List.getElem?_cons_zero,List.getElem?_cons_succ,
    Option.getD_some] at *
  simp only [outputStrides,Strides.at] at ha hb
  omega

theorem finish_inputs_frame (B : Nat) (c : Context) (o : OutputLayout B c)
    (allocated : TensorDescriptor) (word : Index B c.NH c.H c.D→HalfWord) (memory : ByteMemory)
    (fp8 : Bool) (p : Pointers B c o) (hp : pointerChecks B c o fp8 p=Bool.true)
    (q : TensorMetadata) (binding : OutputBinding B c o p q allocated)
    (objects : InputObjects B c o fp8 p) (fresh : FreshInputs objects allocated)
    (t : TensorDescriptor) (ht : t∈[objects.query,objects.key,objects.value])
    (address : Fin (2^64)) (hlo : t.allocationBase≤address.val)
    (hhi : address.val<t.allocationBase+t.allocationBytes) :
    (finishBytes B c o allocated word memory).memory address=memory address := by
  apply finish_frame
  intro i
  have ho:=output_within_allocation B c o fp8 p hp q allocated binding i
  have hf:=fresh t ht
  omega

#axiomsClean actual_inputs_bind
#axiomsClean output_within_allocation
#axiomsClean finish_inputs_frame
end VeriTile.Bench.Repairs.AttentionLlama.Checked
