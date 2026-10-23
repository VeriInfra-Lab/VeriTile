import bench.repairs.tritonbench_g.triton_attention.Contract

namespace VeriTile.Bench.Repairs.TritonAttention.Whole001.Checked
open Native
noncomputable section
open scoped Classical
set_option maxHeartbeats 2500000
deriving instance Fintype for Buffer

abbrev Byte := BitVec 8
abbrev ByteMemory := Fin (2^64)→Byte
def extendMemory (memory : ByteMemory) (n : Nat) : Byte :=
  if h : n<2^64 then memory ⟨n,h⟩ else 0
def encode (word : Word) (j : Fin 4) : Byte := word.extractLsb' (8*j.val) 8
def decode (bytes : Fin 4→Byte) : Word := bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0
def readWord (memory : ByteMemory) (base : Nat) := decode (fun j=>extendMemory memory (base+j.val))

theorem decode_encode (word : Word) : decode (encode word)=word := by
  change (word.extractLsb' 24 8 ++ word.extractLsb' 16 8 ++ word.extractLsb' 8 8 ++ word.extractLsb' 0 8)=word
  rw [BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 24=16+8),
    BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 16=8+8),
    BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 8=0+8)]
  exact BitVec.extractLsb'_eq_self

theorem encode_decode (bytes : Fin 4→Byte) : encode (decode bytes)=bytes := by
  funext j
  fin_cases j
  · change ((bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0) : Word).extractLsb' 0 8=bytes 0
    exact BitVec.extractLsb'_append_eq_right
  · change ((bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0) : Word).extractLsb' 8 8=bytes 1
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8≤8)]
    exact BitVec.extractLsb'_append_eq_right
  · change ((bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0) : Word).extractLsb' 16 8=bytes 2
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8≤16),BitVec.extractLsb'_append_eq_of_le (by decide : 8≤8)]
    exact BitVec.extractLsb'_append_eq_right
  · change ((bytes 3 ++ bytes 2 ++ bytes 1 ++ bytes 0) : Word).extractLsb' 24 8=bytes 3
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8≤24),BitVec.extractLsb'_append_eq_of_le (by decide : 8≤16)]
    exact BitVec.extractLsb'_append_eq_left

def bufferSize (c : Shape) : Buffer→Nat
  | .l | .m | .delta=>c.groups*c.rows
  | .other=>0
  | _=>c.elements
def isRows : Buffer→Bool
  | .l | .m | .delta=>Bool.true
  | _=>Bool.false

/-- Ordinary contiguous storage may retain arbitrary singleton strides. -/
def PackedStrides (c : Shape) (t : Tensor) : Prop :=
  (c.batches=1 ∨ t.observation.strides[0]?.getD 0=c.heads*c.rows*c.dim) ∧
  (c.heads=1 ∨ t.observation.strides[1]?.getD 0=c.rows*c.dim) ∧
  t.observation.strides[2]?.getD 0=c.dim ∧ t.observation.strides[3]?.getD 0=1

theorem canonical_packed (c : Shape) (t : Tensor) (h : t.observation.strides=Host.strides c) :
    PackedStrides c t := by simp [PackedStrides,h,Host.strides]

/-- Actual source object metadata. Input metadata is checked; allocation and
the original contiguous-copy metadata use ordinary Tensor interpretation. -/
structure Objects (c : Shape) (device : Nat) where
  tensor : Buffer→Tensor
  metadata : ∀ r,r≠Buffer.other → (tensor r).observation=
    if isRows r then Host.rowTensor c device r else
      {Host.tensor c device r with strides:=(tensor r).observation.strides}
  packed : ∀ r,r≠Buffer.other → isRows r=Bool.false → PackedStrides c (tensor r)
  width : ∀ r,r≠Buffer.other → (tensor r).width=4
  count : ∀ r,(tensor r).numel=bufferSize c r

def callerWords (objects : Buffer→Tensor) (memory : ByteMemory) : Memory Word :=
  fun r n=>readWord memory ((objects r).dataPointer+4*n)

abbrev ByteIndex (c : Shape) (writes : List Buffer) :=
  (r : {r : Buffer // r∈writes}) × (Fin (bufferSize c r.val) × Fin 4)
def byteAddress (c : Shape) (objects : Buffer→Tensor) (writes : List Buffer) (i : ByteIndex c writes) :=
  (objects i.1.val).dataPointer+4*i.2.1.val+i.2.2.val

def Disjoint (a b : Tensor) :=
  a.storageBase+a.storageBytes≤b.storageBase ∨ b.storageBase+b.storageBytes≤a.storageBase
def SeparateWrites (objects : Buffer→Tensor) (writes : List Buffer) :=
  ∀ r∈writes,∀ s∈writes,r≠s → Disjoint (objects r) (objects s)

theorem byte_in_allocation (g : Geometry) (hs : shapeChecks g=Bool.true)
    (device : Nat) (objects : Objects (checkedShape g hs) device) (writes : List Buffer)
    (noOther : Buffer.other∉writes) (hp : ∀ r∈writes,pointerChecks g (objects.tensor r)=Bool.true)
    (i : ByteIndex (checkedShape g hs) writes) :
    (objects.tensor i.1.val).storageBase≤byteAddress _ objects.tensor writes i ∧
    byteAddress _ objects.tensor writes i<(objects.tensor i.1.val).storageBase+(objects.tensor i.1.val).storageBytes := by
  have hr : i.1.val≠Buffer.other := fun h=>noOther (h ▸ i.1.property)
  have bound:=active_bytes g (objects.tensor i.1.val) (hp _ i.1.property) i.2.1.val
    (by rw [objects.count];exact i.2.1.isLt) i.2.2.val (by rw [objects.width _ hr];exact i.2.2.isLt)
  simpa only [byteAddress,objects.width _ hr] using bound

theorem byteAddress_injective (g : Geometry) (hs : shapeChecks g=Bool.true)
    (device : Nat) (objects : Objects (checkedShape g hs) device) (writes : List Buffer)
    (noOther : Buffer.other∉writes) (hp : ∀ r∈writes,pointerChecks g (objects.tensor r)=Bool.true)
    (fresh : SeparateWrites objects.tensor writes) :
    Function.Injective (byteAddress (checkedShape g hs) objects.tensor writes) := by
  intro x y he
  have hx:=byte_in_allocation g hs device objects writes noOther hp x
  have hy:=byte_in_allocation g hs device objects writes noOther hp y
  have hr : x.1.val=y.1.val := by
    by_contra hn
    have hf:=fresh _ x.1.property _ y.1.property hn
    unfold Disjoint at hf
    omega
  rcases x with ⟨⟨r,hrmem⟩,i,j⟩
  rcases y with ⟨⟨s,hsmem⟩,k,l⟩
  change r=s at hr
  subst s
  have hj:=j.isLt
  have hl:=l.isLt
  simp only [byteAddress] at he
  have hi : i=k := Fin.ext (by omega)
  have hbyte : j=l := Fin.ext (by omega)
  cases hi
  cases hbyte
  rfl

theorem bufferSize_bound (c : Shape) (r : Buffer) : bufferSize c r≤c.elements := by
  have hd:=c.dim_pos
  have hm:=Nat.mul_le_mul_left (c.groups*c.rows) hd
  cases r <;> simp only [bufferSize] <;> try omega
  all_goals simpa only [Shape.elements,Nat.mul_one] using hm

theorem byte_bounded (g : Geometry) (hs : shapeChecks g=Bool.true)
    (device : Nat) (objects : Objects (checkedShape g hs) device) (writes : List Buffer)
    (noOther : Buffer.other∉writes) (hp : ∀ r∈writes,pointerChecks g (objects.tensor r)=Bool.true)
    (i : ByteIndex (checkedShape g hs) writes) : byteAddress _ objects.tensor writes i<2^64 := by
  have hr : i.1.val≠Buffer.other := fun h=>noOther (h ▸ i.1.property)
  have hb:=bufferSize_bound (checkedShape g hs) i.1.val
  have hn:=i.2.1.isLt
  have he : (checkedShape g hs).elements=g.batches*g.heads*g.rows*g.dim := by
    simp only [Shape.elements,Shape.groups,checked_rows]
    rfl
  have hf:=formed_bytes g hs (objects.tensor i.1.val) (hp _ i.1.property) i.2.1.val
    (by rw [he] at hb;unfold pointerEnvelope;omega) i.2.2.val (by rw [objects.width _ hr];exact i.2.2.isLt)
  simpa only [byteAddress,objects.width _ hr] using hf.2.2

def storeAll {ι : Type} (address : ι→Nat) (word : ι→Byte) (memory : Nat→Byte) (schedule : List ι) : Nat→Byte :=
  schedule.foldl (fun mem i=>Function.update mem (address i) (word i)) memory

private theorem store_read {ι : Type} [DecidableEq ι] (address : ι→Nat) (word : ι→Byte)
    (hinj : Function.Injective address) (schedule : List ι) (memory : Nat→Byte) (i : ι) :
    storeAll address word memory schedule (address i)=if i∈schedule then word i else memory (address i) := by
  induction schedule generalizing memory with
  | nil=>simp [storeAll]
  | cons j rest ih=>
    simp only [storeAll,List.foldl_cons]
    change storeAll address word (Function.update memory (address j) (word j)) rest (address i)=_
    rw [ih]
    by_cases hr : i∈rest
    · simp [hr]
    · by_cases he : i=j
      · subst j;simp [hr]
      · have hn : address i≠address j := fun h=>he (hinj h)
        simp [hr,he,hn]

private theorem store_frame {ι : Type} (address : ι→Nat) (word : ι→Byte) (schedule : List ι)
    (memory : Nat→Byte) (location : Nat) (h : ∀ i∈schedule,location≠address i) :
    storeAll address word memory schedule location=memory location := by
  induction schedule generalizing memory with
  | nil=>rfl
  | cons j rest ih=>
    simp only [storeAll,List.foldl_cons]
    change storeAll address word (Function.update memory (address j) (word j)) rest location=_
    rw [ih _ (fun i hi=>h i (List.mem_cons_of_mem j hi))]
    exact Function.update_of_ne (h j List.mem_cons_self) ..

/-- Ordinary four-byte lowering of the complete source stores. The caller's
initial bytes outside the exact stage write window are retained verbatim. -/
def finishBytes (c : Shape) (objects : Buffer→Tensor) (writes : List Buffer)
    (after : Memory Word) (before : ByteMemory) : ByteMemory := fun location=>
  storeAll (byteAddress c objects writes) (fun i=>encode (after i.1.val i.2.1.val) i.2.2)
    (extendMemory before) Finset.univ.toList location.val

theorem finish_byte (g : Geometry) (hs : shapeChecks g=Bool.true)
    (device : Nat) (objects : Objects (checkedShape g hs) device) (writes : List Buffer)
    (noOther : Buffer.other∉writes) (hp : ∀ r∈writes,pointerChecks g (objects.tensor r)=Bool.true)
    (fresh : SeparateWrites objects.tensor writes) (after : Memory Word) (before : ByteMemory)
    (i : ByteIndex (checkedShape g hs) writes) :
    extendMemory (finishBytes (checkedShape g hs) objects.tensor writes after before) (byteAddress _ objects.tensor writes i)=
      encode (after i.1.val i.2.1.val) i.2.2 := by
  rw [extendMemory,dif_pos (byte_bounded g hs device objects writes noOther hp i)]
  exact (store_read _ _ (byteAddress_injective g hs device objects writes noOther hp fresh)
    Finset.univ.toList (extendMemory before) i).trans (if_pos (by simp))

theorem finish_word (g : Geometry) (hs : shapeChecks g=Bool.true)
    (device : Nat) (objects : Objects (checkedShape g hs) device) (writes : List Buffer)
    (noOther : Buffer.other∉writes) (hp : ∀ r∈writes,pointerChecks g (objects.tensor r)=Bool.true)
    (fresh : SeparateWrites objects.tensor writes) (after : Memory Word) (before : ByteMemory)
    (r : Buffer) (hr : r∈writes) (n : Fin (bufferSize (checkedShape g hs) r)) :
    readWord (finishBytes (checkedShape g hs) objects.tensor writes after before) ((objects.tensor r).dataPointer+4*n.val)=after r n.val := by
  unfold readWord
  have he:=funext (fun j=>finish_byte g hs device objects writes noOther hp fresh after before ⟨⟨r,hr⟩,n,j⟩)
  change decode (fun j=>extendMemory _ (byteAddress _ objects.tensor writes ⟨⟨r,hr⟩,n,j⟩))=_
  rw [he]
  exact decode_encode (after r n.val)

theorem finish_frame (c : Shape) (objects : Buffer→Tensor) (writes : List Buffer)
    (after : Memory Word) (before : ByteMemory) (location : Fin (2^64))
    (outside : ∀ i : ByteIndex c writes,location.val≠byteAddress c objects writes i) :
    finishBytes c objects writes after before location=before location := by
  change storeAll _ _ _ _ location.val=_
  rw [store_frame _ _ _ _ location.val (fun i _=>outside i)]
  exact dif_pos location.isLt

theorem finish_preserves_allocation (g : Geometry) (hs : shapeChecks g=Bool.true)
    (device : Nat) (objects : Objects (checkedShape g hs) device) (writes : List Buffer)
    (noOther : Buffer.other∉writes) (hp : ∀ r∈writes,pointerChecks g (objects.tensor r)=Bool.true)
    (after : Memory Word) (before : ByteMemory) (input : Tensor)
    (fresh : ∀ r∈writes,Disjoint (objects.tensor r) input) (location : Fin (2^64))
    (hlo : input.storageBase≤location.val) (hhi : location.val < input.storageBase+input.storageBytes) :
    finishBytes (checkedShape g hs) objects.tensor writes after before location=before location := by
  apply finish_frame
  intro i
  have hb:=byte_in_allocation g hs device objects writes noOther hp i
  have hf:=fresh _ i.1.property
  unfold Disjoint at hf
  omega

#axiomsClean decode_encode
#axiomsClean encode_decode
#axiomsClean byte_in_allocation
#axiomsClean byteAddress_injective
#axiomsClean bufferSize_bound
#axiomsClean byte_bounded
#axiomsClean finish_byte
#axiomsClean finish_word
#axiomsClean finish_frame
#axiomsClean finish_preserves_allocation
end
end VeriTile.Bench.Repairs.TritonAttention.Whole001.Checked
