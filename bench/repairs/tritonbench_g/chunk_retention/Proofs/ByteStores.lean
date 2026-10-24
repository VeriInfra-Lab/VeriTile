import bench.repairs.tritonbench_g.chunk_retention.Contract
import Mathlib.Tactic.FinCases
import Mathlib.Data.Fintype.Fin

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.Checked
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000
abbrev Word := BitVec 32
abbrev Byte := BitVec 8
abbrev ByteMemory := Fin (2^64)→Byte

def extendMemory (m : ByteMemory) (n : Nat) : Byte := if h:n<2^64 then m ⟨n,h⟩ else 0
def encode (w : Word) (j : Fin 4) : Byte := w.extractLsb' (8*j.val) 8
def decode (b : Fin 4→Byte) : Word := b 3 ++ b 2 ++ b 1 ++ b 0
def readWord (m : ByteMemory) (base : Nat) := decode (fun j=>extendMemory m (base+j.val))

theorem decode_encode (w : Word) : decode (encode w)=w := by
  change (w.extractLsb' 24 8 ++ w.extractLsb' 16 8 ++ w.extractLsb' 8 8 ++ w.extractLsb' 0 8)=w
  rw [BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 24=16+8),
    BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 16=8+8),
    BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 8=0+8)]
  exact BitVec.extractLsb'_eq_self

theorem encode_decode (b : Fin 4→Byte) : encode (decode b)=b := by
  funext j
  fin_cases j
  · change ((b 3 ++ b 2 ++ b 1 ++ b 0) : Word).extractLsb' 0 8=b 0
    exact BitVec.extractLsb'_append_eq_right
  · change ((b 3 ++ b 2 ++ b 1 ++ b 0) : Word).extractLsb' 8 8=b 1
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8≤8)]
    exact BitVec.extractLsb'_append_eq_right
  · change ((b 3 ++ b 2 ++ b 1 ++ b 0) : Word).extractLsb' 16 8=b 2
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8≤16),BitVec.extractLsb'_append_eq_of_le (by decide : 8≤8)]
    exact BitVec.extractLsb'_append_eq_right
  · change ((b 3 ++ b 2 ++ b 1 ++ b 0) : Word).extractLsb' 24 8=b 3
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8≤24),BitVec.extractLsb'_append_eq_of_le (by decide : 8≤16)]
    exact BitVec.extractLsb'_append_eq_left

private theorem digits3 (B C a b d a' b' d' : Nat) (hb : b<B) (hb' : b'<B) (hd : d<C) (hd' : d'<C)
    (he : (a*B+b)*C+d=(a'*B+b')*C+d') : a=a' ∧ b=b' ∧ d=d' := by
  have h:=digit_injective C _ _ _ _ hd hd' he
  have h':=digit_injective B _ _ _ _ hb hb' h.1
  exact ⟨h'.1,h'.2,h.2⟩

private theorem digits4 (B C D a b d e a' b' d' e' : Nat)
    (hb : b<B) (hb' : b'<B) (hd : d<C) (hd' : d'<C) (he : e<D) (he' : e'<D)
    (eq : ((a*B+b)*C+d)*D+e=((a'*B+b')*C+d')*D+e') : a=a' ∧ b=b' ∧ d=d' ∧ e=e' := by
  have h:=digit_injective D _ _ _ _ he he' eq
  have h':=digits3 B C a b d a' b' d' hb hb' hd hd' h.1
  exact ⟨h'.1,h'.2.1,h'.2.2,h.2⟩

theorem offset_injective (c : Shape) (a b : Address) (ha : a.Valid c) (hb : b.Valid c)
    (region : a.region=b.region) (offset : a.offset c=b.offset c) : a=b := by
  rcases a with ⟨r,h,s,i,j⟩
  rcases b with ⟨r',h',s',i',j'⟩
  simp only at region
  subst r'
  cases r <;> simp only [Address.Valid] at ha hb
  all_goals simp only [Address.offset,Shape.keyOffset,Shape.keyHead,Shape.valueOffset,Shape.valueHead,
    Shape.stateOffset,Shape.stateHead,Shape.historyOffset,Shape.historyHead,Shape.partialOffset,
    Shape.valueSize,Nat.mul_assoc,Nat.add_assoc] at offset
  all_goals first
    | (have hcoords:=digits3 c.T c.K h i j h' i' j' ha.2.2.1 hb.2.2.1 ha.2.2.2 hb.2.2.2
        (by simpa only [Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using offset)
       rcases ha.2.1 with rfl;rcases hb.2.1 with rfl
       rcases hcoords with ⟨rfl,rfl,rfl⟩;rfl)
    | (have hcoords:=digits3 c.T c.V h i j h' i' j' ha.2.2.1 hb.2.2.1 ha.2.2.2 hb.2.2.2
        (by simpa only [Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using offset)
       rcases ha.2.1 with rfl;rcases hb.2.1 with rfl
       rcases hcoords with ⟨rfl,rfl,rfl⟩;rfl)
    | (have hcoords:=digits3 c.K c.V h i j h' i' j' ha.2.2.1 hb.2.2.1 ha.2.2.2 hb.2.2.2
        (by simpa only [Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using offset)
       rcases ha.2.1 with rfl;rcases hb.2.1 with rfl
       rcases hcoords with ⟨rfl,rfl,rfl⟩;rfl)
    | (have hcoords:=digits4 c.chunks c.K c.V h s i j h' s' i' j'
        ha.2.1 hb.2.1 ha.2.2.1 hb.2.2.1 ha.2.2.2 hb.2.2.2
        (by simpa only [Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using offset)
       rcases hcoords with ⟨rfl,rfl,rfl,rfl⟩;rfl)
    | (have hcoords:=digits4 c.heads c.T c.V s h i j s' h' i' j'
        ha.1 hb.1 ha.2.2.1 hb.2.2.1 ha.2.2.2 hb.2.2.2
        (by simpa only [Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using offset)
       rcases hcoords with ⟨rfl,rfl,rfl,rfl⟩;rfl)

def Disjoint (a b : Tensor) := a.storageBase+a.storageBytes≤b.storageBase ∨ b.storageBase+b.storageBytes≤a.storageBase

def cellAddress (c : Shape) (objects : Region→Tensor) (a : Address) := (objects a.region).dataPointer+4*a.offset c

def byteAddress (c : Shape) (objects : Region→Tensor) (i : Address×Fin 4) := cellAddress c objects i.1+i.2.val

def byteEvents (events : List Address) := events.flatMap (fun a=>(List.finRange 4).map (fun j=>(a,j)))

@[simp] theorem byteEvents_mem (events : List Address) (a : Address) (j : Fin 4) :
    (a,j)∈byteEvents events ↔ a∈events := by simp [byteEvents]

/-- Executed pointer checks and ordinary observed allocation shapes. -/
structure StoreAdmission (c : Shape) (objects : Region→Tensor) (events : List Address) : Prop where
  valid : ∀ a∈events,a.Valid c
  checked : ∀ a∈events,pointerChecks (extent c a.region) (objects a.region)=Bool.true
  count : ∀ a∈events,(objects a.region).numel=a.region.capacity c

/-- Fresh ordinary allocations need only separate the buffers actually written
by this call. No input/input separation or numerical premise occurs here. -/
def SeparateStores (objects : Region→Tensor) (events : List Address) :=
  ∀ a∈events,∀ b∈events,a.region≠b.region → Disjoint (objects a.region) (objects b.region)

theorem byte_bounds (c : Shape) (objects : Region→Tensor) (events : List Address)
    (checks : StoreAdmission c objects events) (i : Address×Fin 4) (hi : i∈byteEvents events) :
    (objects i.1.region).storageBase≤byteAddress c objects i ∧
    byteAddress c objects i<(objects i.1.region).storageBase+(objects i.1.region).storageBytes ∧
    byteAddress c objects i<2^64 := by
  have ha : i.1∈events := (byteEvents_mem events i.1 i.2).mp hi
  exact active_bytes _ _ (checks.checked _ ha) (i.1.offset c)
    (by rw [checks.count _ ha];exact Address.offset_bound c i.1 (checks.valid _ ha)) i.2.val i.2.isLt

theorem byte_injective (c : Shape) (objects : Region→Tensor) (events : List Address)
    (checks : StoreAdmission c objects events) (fresh : SeparateStores objects events)
    (i j : Address×Fin 4) (hi : i∈byteEvents events) (hj : j∈byteEvents events)
    (he : byteAddress c objects i=byteAddress c objects j) : i=j := by
  have hai : i.1∈events := (byteEvents_mem events i.1 i.2).mp hi
  have haj : j.1∈events := (byteEvents_mem events j.1 j.2).mp hj
  have bi:=byte_bounds c objects events checks i hi
  have bj:=byte_bounds c objects events checks j hj
  have region : i.1.region=j.1.region := by
    by_contra hn
    have hf:=fresh _ hai _ haj hn
    unfold Disjoint at hf
    omega
  have hli:=i.2.isLt
  have hlj:=j.2.isLt
  unfold byteAddress cellAddress at he
  rw [region] at he
  have hoff : i.1.offset c=j.1.offset c := by omega
  have ha:=offset_injective c i.1 j.1 (checks.valid _ hai) (checks.valid _ haj) region hoff
  exact Prod.ext ha (Fin.ext (by omega))

def storeAll {ι α : Type} (address : ι→Nat) (value : ι→α) (before : Nat→α) (schedule : List ι) : Nat→α :=
  schedule.foldl (fun memory i=>Function.update memory (address i) (value i)) before

theorem store_keeps {ι α : Type} (address : ι→Nat) (value : ι→α)
    (schedule : List ι) (before : Nat→α) (point : Nat) (v : α)
    (initial : before point=v) (consistent : ∀ j∈schedule,address j=point → value j=v) :
    storeAll address value before schedule point=v := by
  induction schedule generalizing before with
  | nil=>exact initial
  | cons j rest ih=>
    simp only [storeAll,List.foldl_cons]
    apply ih
    · by_cases he : point=address j
      · rw [he,Function.update_self,consistent j (by simp) he.symm]
      · rw [Function.update_of_ne he];exact initial
    · intro k hk;exact consistent k (List.mem_cons_of_mem _ hk)

theorem store_read {ι α : Type} [DecidableEq ι] (address : ι→Nat) (value : ι→α)
    (schedule : List ι) (before : Nat→α) (i : ι) (hi : i∈schedule)
    (consistent : ∀ j∈schedule,address j=address i → value j=value i) :
    storeAll address value before schedule (address i)=value i := by
  induction schedule generalizing before with
  | nil=>simp at hi
  | cons j rest ih=>
    simp only [storeAll,List.foldl_cons]
    change storeAll address value (Function.update before (address j) (value j)) rest (address i)=_
    by_cases hr : i∈rest
    · exact ih _ hr (fun j hj=>consistent j (List.mem_cons_of_mem _ hj))
    · have he : i=j := (List.mem_cons.mp hi).resolve_right hr
      subst j
      exact store_keeps address value rest _ (address i) (value i) (Function.update_self ..)
        (fun j hj=>consistent j (List.mem_cons_of_mem _ hj))

theorem store_frame {ι α : Type} (address : ι→Nat) (value : ι→α) (schedule : List ι)
    (before : Nat→α) (point : Nat) (outside : ∀ i∈schedule,point≠address i) :
    storeAll address value before schedule point=before point := by
  induction schedule generalizing before with
  | nil=>rfl
  | cons i rest ih=>
    simp only [storeAll,List.foldl_cons]
    change storeAll address value (Function.update before (address i) (value i)) rest point=_
    rw [ih _ (fun j hj=>outside j (List.mem_cons_of_mem _ hj))]
    exact Function.update_of_ne (outside i List.mem_cons_self) ..

def finishBytes (c : Shape) (objects : Region→Tensor) (events : List Address)
    (after : Heap Word) (before : ByteMemory) : ByteMemory := fun point=>
  storeAll (byteAddress c objects) (fun i=>encode (after i.1) i.2) (extendMemory before) (byteEvents events) point.val

theorem finish_byte (c : Shape) (objects : Region→Tensor) (events : List Address)
    (checks : StoreAdmission c objects events) (fresh : SeparateStores objects events)
    (after : Heap Word) (before : ByteMemory) (a : Address) (ha : a∈events) (j : Fin 4) :
    extendMemory (finishBytes c objects events after before) (byteAddress c objects (a,j))=encode (after a) j := by
  have hi : (a,j)∈byteEvents events := (byteEvents_mem events a j).mpr ha
  rw [extendMemory,dif_pos (byte_bounds c objects events checks (a,j) hi).2.2]
  exact store_read _ _ _ _ (a,j) hi (fun b hb he=>congrArg (fun i=>encode (after i.1) i.2)
    (byte_injective c objects events checks fresh b (a,j) hb hi he))

theorem finish_word (c : Shape) (objects : Region→Tensor) (events : List Address)
    (checks : StoreAdmission c objects events) (fresh : SeparateStores objects events)
    (after : Heap Word) (before : ByteMemory) (a : Address) (ha : a∈events) :
    readWord (finishBytes c objects events after before) (cellAddress c objects a)=after a := by
  unfold readWord
  have h:=funext (fun j=>finish_byte c objects events checks fresh after before a ha j)
  change decode (fun j=>extendMemory _ (byteAddress c objects (a,j)))=_
  rw [h]
  exact decode_encode _

theorem finish_frame (c : Shape) (objects : Region→Tensor) (events : List Address)
    (after : Heap Word) (before : ByteMemory) (point : Fin (2^64))
    (outside : ∀ i∈byteEvents events,point.val≠byteAddress c objects i) :
    finishBytes c objects events after before point=before point := by
  change storeAll _ _ _ _ point.val=_
  rw [store_frame _ _ _ _ _ outside]
  exact dif_pos point.isLt

#axiomsClean decode_encode
#axiomsClean encode_decode
#axiomsClean offset_injective
#axiomsClean byte_bounds
#axiomsClean byte_injective
#axiomsClean store_read
#axiomsClean store_frame
#axiomsClean finish_byte
#axiomsClean finish_word
#axiomsClean finish_frame
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.Checked
