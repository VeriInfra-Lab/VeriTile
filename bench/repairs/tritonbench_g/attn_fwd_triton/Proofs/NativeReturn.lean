import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.CheckedFPEquiv
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.OutputTag

/-! Actual FP32 input bytes and the BF16 returned Tensor. The byte stores below
are the ordinary little-endian lowering of the complete source grid's disjoint
BF16 stores. Local native operations retain the selected source precision. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Checked
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.LayerNormFwd
open Verified.PhysicalCells (Role Layout width FreshBytes)
open TypedPhysical (Physical pull)
open TypedExecution NativeGraph
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000

abbrev Byte := BitVec 8
abbrev BF16Word := BitVec 16
abbrev ByteMemory := Fin (2^64)→Byte

def extendMemory (memory : ByteMemory) (n : Nat) : Byte :=
  if h : n<2^64 then memory ⟨n,h⟩ else 0

def readFP32 (memory : ByteMemory) (n : Nat) : Word := BitVec.ofNat 32
  ((extendMemory memory n).toNat+256*(extendMemory memory (n+1)).toNat+
    65536*(extendMemory memory (n+2)).toNat+16777216*(extendMemory memory (n+3)).toNat)

def splitBF16 (w : BF16Word) (j : Fin 2) : Byte := BitVec.ofNat 8 (w.toNat/256^j.val)
def joinBF16 (bytes : Fin 2→Byte) : BF16Word :=
  BitVec.ofNat 16 ((bytes 0).toNat+256*(bytes 1).toNat)
def readBF16 (memory : ByteMemory) (n : Nat) : BF16Word :=
  joinBF16 (fun j=>extendMemory memory (n+j.val))

theorem join_split (w : BF16Word) : joinBF16 (splitBF16 w)=w := by
  apply BitVec.eq_of_toNat_eq
  have hw:=w.isLt
  simp only [joinBF16,splitBF16,BitVec.toNat_ofNat,Fin.val_zero,Fin.val_one,pow_zero,pow_one,Nat.div_one]
  norm_num at hw ⊢
  omega

/-- All input ports are actual FP32 loads. Output is fresh and overwritten, so
its initial tag is immaterial. The full caller byte memory remains separate. -/
def callerCells (a : Layout) (memory : ByteMemory) : Physical Word a.toLayout :=
  fun cell=>Cell.mk .fp32 (readFP32 memory cell.val)

/-- Ordinary Tensor observations: identity, allocation, native dtype and the
same metadata observed by the host checks. No output or access bound is a field. -/
structure TensorDescriptor where
  (object storage dataPointer allocationBase allocationBytes : Nat)
  dtype : TileDType
  metadata : TensorMetadata
  metadataType : metadata.fp32=decide (dtype=.fp32)

def pointers (objects : Role→TensorDescriptor) (r : Role) := (objects r).dataPointer
def descriptorElement (t : TensorDescriptor) (z h n d : Nat) :=
  z*(t.metadata.strides[0]?.getD 0).toNat+h*(t.metadata.strides[1]?.getD 0).toNat+
    n*(t.metadata.strides[2]?.getD 0).toNat+d*(t.metadata.strides[3]?.getD 0).toNat

def descriptorRead (objects : Role→TensorDescriptor) (memory : ByteMemory)
    (r : Role) (z h n d : Nat) : Word :=
  readFP32 memory ((objects r).dataPointer+4*(match r with
    | .queryScale | .keyScale=>n
    | _=>descriptorElement (objects r) z h n d))

/-- The graph's original memory is read from the actual caller objects, with
independent strides for Q, K and V and packed contiguous scale indexing. -/
theorem caller_inputs (p : Config) (a : Layout) (objects : Role→TensorDescriptor)
    (views : Views a p (pointers objects)) (memory : ByteMemory)
    (hinput : inputChecks (typedGeometry p) (objects .query).metadata (objects .key).metadata
      (objects .value).metadata (objects .queryScale).metadata (objects .keyScale).metadata=Bool.true)
    (z h n d : Nat) (r : Role) (hr : r≠Role.output) (ha : roleActive p z h n d r) :
    TypedPhysical.read a.toLayout (callerCells a memory) (roleKey p z h n d r)=
      Cell.mk .fp32 (descriptorRead objects memory r z h n d) := by
  have hl:=views.live z h n d r ha
  have hp:=views.address z h n d r ha
  simp only [TypedPhysical.read,dif_pos hl,callerCells]
  rw [hp]
  simp only [inputChecks,Bool.and_eq_true,decide_eq_true_eq] at hinput
  have hq:=hinput.2.2.2.2.1
  have hk:=hinput.2.2.2.2.2.1
  have hv:=hinput.2.2.2.2.2.2
  cases r <;> try contradiction
  all_goals simp [descriptorRead,descriptorElement,pointers,roleOffset,width,hq,hk,hv,
    typedGeometry,Strides.raw]

structure OutputBinding (p : Config) (objects : Role→TensorDescriptor) : Prop where
  dtype : (objects .output).dtype=.bf16
  shape : (objects .output).metadata.shape=[p.batches,p.heads,p.context,p.headDim]
  strides : (objects .output).metadata.strides=((typedGeometry p).stride .output).raw
  device : (objects .output).metadata.device=(objects .query).metadata.device
  lower : (objects .output).allocationBase≤(objects .output).dataPointer
  backed : (objects .output).dataPointer+2*span (typedGeometry p) .output≤
    (objects .output).allocationBase+(objects .output).allocationBytes

abbrev ByteIndex (p : Config) := Coordinate p × Fin 2
def outputByte (p : Config) (objects : Role→TensorDescriptor) (i : ByteIndex p) :=
  (objects .output).dataPointer+2*logicalAddress p i.1+i.2.val

theorem outputByte_injective (p : Config) (objects : Role→TensorDescriptor) (hn : NonOverlapping p) :
    Function.Injective (outputByte p objects) := by
  intro x y he
  have hx:=x.2.isLt
  have hy:=y.2.isLt
  have hc : logicalAddress p x.1=logicalAddress p y.1 := by unfold outputByte at he; omega
  apply Prod.ext (hn hc)
  apply Fin.ext
  unfold outputByte at he
  rw [hc] at he
  omega

theorem outputByte_bounded (p : Config) (objects : Role→TensorDescriptor)
    (hp : pointerChecks (typedGeometry p) (pointers objects)=Bool.true) (i : ByteIndex p) :
    outputByte p objects i<2^64 := by
  have hb:=matrix_formed (typedGeometry p) .output (by decide) (by decide)
    i.1.1.val i.1.2.1.val i.1.2.2.1.val i.1.2.2.2.val
    (Nat.le_of_lt i.1.1.isLt) (Nat.le_of_lt i.1.2.1.isLt)
    (by have := i.1.2.2.1.isLt; change _≤p.context+256; omega)
    (Nat.le_trans (Nat.le_of_lt i.1.2.2.2.isLt) (HostShape.blockD_covers _))
  have hf:=formed_pointer (typedGeometry p) (pointers objects) hp .output _ hb
  have hj:=i.2.isLt
  change _ ∧ (objects .output).dataPointer+2*logicalAddress p i.1+2<2^64 ∧ _ at hf
  unfold outputByte
  omega

theorem outputByte_descriptor (p : Config) (objects : Role→TensorDescriptor)
    (binding : OutputBinding p objects) (i : ByteIndex p) :
    outputByte p objects i=(objects .output).dataPointer+
      2*descriptorElement (objects .output) i.1.1.val i.1.2.1.val i.1.2.2.1.val i.1.2.2.2.val+i.2.val := by
  simp [outputByte,descriptorElement,binding.strides,typedGeometry,Strides.raw,logicalAddress]

/-- Every serialized element is exactly an active store of the actual grid. -/
theorem byte_source_store (p : Config) (i : ByteIndex p) :
    Written p (HostShape.blockD p.headDim) (owner p i.1).1.val (owner p i.1).2.val
      p.Out (logicalAddress p i.1) := covered p _ (HostShape.blockD_covers _) i.1

def storeBytes {ι : Type} (address : ι→Nat) (value : ι→Byte) (memory : Nat→Byte)
    (schedule : List ι) : Nat→Byte :=
  schedule.foldl (fun mem i=>Function.update mem (address i) (value i)) memory

private theorem store_read {ι : Type} [DecidableEq ι] (address : ι→Nat) (value : ι→Byte)
    (hinj : Function.Injective address) (schedule : List ι) (memory : Nat→Byte) (i : ι) :
    storeBytes address value memory schedule (address i)=
      if i∈schedule then value i else memory (address i) := by
  induction schedule generalizing memory with
  | nil=>simp [storeBytes]
  | cons j rest ih=>
    simp only [storeBytes,List.foldl_cons]
    change storeBytes address value (Function.update memory (address j) (value j)) rest (address i)=_
    rw [ih]
    by_cases hr : i∈rest
    · simp [hr]
    · by_cases he : i=j
      · subst j; simp [hr]
      · have hn : address i≠address j := fun h=>he (hinj h)
        simp [hr,he,hn]

private theorem store_frame {ι : Type} (address : ι→Nat) (value : ι→Byte)
    (schedule : List ι) (memory : Nat→Byte) (location : Nat)
    (h : ∀ i∈schedule,location≠address i) :
    storeBytes address value memory schedule location=memory location := by
  induction schedule generalizing memory with
  | nil=>rfl
  | cons j rest ih=>
    simp only [storeBytes,List.foldl_cons]
    change storeBytes address value (Function.update memory (address j) (value j)) rest location=_
    rw [ih _ (fun i hi=>h i (List.mem_cons_of_mem j hi))]
    exact Function.update_of_ne (h j List.mem_cons_self) ..

def byteSchedule (p : Config) : List (ByteIndex p) := Finset.univ.toList

structure ReturnedBytes where
  tensor : TensorDescriptor
  memory : ByteMemory

/-- Ordinary BF16 store encoding uses the low sixteen bits of the native port.
Disjoint source stores may be serialized in this finite coordinate order. -/
def finishBytes (p : Config) (objects : Role→TensorDescriptor)
    (word : Coordinate p→BF16Word) (memory : ByteMemory) : ReturnedBytes :=
  ⟨objects .output,fun address=>storeBytes (outputByte p objects) (fun i=>splitBF16 (word i.1) i.2)
    (extendMemory memory) (byteSchedule p) address.val⟩

def ReturnedBytes.read (p : Config) (result : ReturnedBytes) (c : Coordinate p) : BF16Word :=
  readBF16 result.memory (result.tensor.dataPointer+
    2*descriptorElement result.tensor c.1.val c.2.1.val c.2.2.1.val c.2.2.2.val)

theorem finish_byte (p : Config) (objects : Role→TensorDescriptor) (hn : NonOverlapping p)
    (word : Coordinate p→BF16Word) (memory : ByteMemory)
    (hp : pointerChecks (typedGeometry p) (pointers objects)=Bool.true) (i : ByteIndex p) :
    (finishBytes p objects word memory).memory
      ⟨outputByte p objects i,outputByte_bounded p objects hp i⟩=splitBF16 (word i.1) i.2 := by
  exact (store_read (outputByte p objects) _ (outputByte_injective p objects hn) (byteSchedule p)
    (extendMemory memory) i).trans (if_pos (by simp [byteSchedule]))

theorem finish_read (p : Config) (objects : Role→TensorDescriptor) (hn : NonOverlapping p)
    (binding : OutputBinding p objects) (word : Coordinate p→BF16Word) (memory : ByteMemory)
    (hp : pointerChecks (typedGeometry p) (pointers objects)=Bool.true) (c : Coordinate p) :
    (finishBytes p objects word memory).read p c=word c := by
  unfold ReturnedBytes.read readBF16
  change joinBF16 (fun j=>extendMemory (finishBytes p objects word memory).memory
    ((objects .output).dataPointer+2*descriptorElement (objects .output)
      c.1.val c.2.1.val c.2.2.1.val c.2.2.2.val+j.val))=_
  have he : (fun j : Fin 2=>extendMemory (finishBytes p objects word memory).memory
      ((objects .output).dataPointer+2*descriptorElement (objects .output)
        c.1.val c.2.1.val c.2.2.1.val c.2.2.2.val+j.val))=splitBF16 (word c) := by
    funext j
    rw [←outputByte_descriptor p objects binding (c,j)]
    rw [extendMemory,dif_pos (outputByte_bounded p objects hp (c,j))]
    exact finish_byte p objects hn word memory hp (c,j)
  rw [he,join_split]

theorem finish_frame (p : Config) (objects : Role→TensorDescriptor)
    (word : Coordinate p→BF16Word) (memory : ByteMemory) (address : Fin (2^64))
    (h : ∀ i,address.val≠outputByte p objects i) :
    (finishBytes p objects word memory).memory address=memory address := by
  change storeBytes _ _ _ _ address.val=_
  rw [store_frame _ _ _ _ address.val (fun i _=>h i)]
  exact dif_pos address.isLt

def FreshInputs (objects : Role→TensorDescriptor) : Prop :=
  ∀ r, r≠Role.output →
    (objects r).allocationBase+(objects r).allocationBytes≤(objects .output).allocationBase ∨
    (objects .output).allocationBase+(objects .output).allocationBytes≤(objects r).allocationBase

theorem output_within_allocation (p : Config) (objects : Role→TensorDescriptor)
    (binding : OutputBinding p objects) (i : ByteIndex p) :
    (objects .output).allocationBase≤outputByte p objects i ∧
    outputByte p objects i<(objects .output).allocationBase+(objects .output).allocationBytes := by
  have ha:=active_bound (typedGeometry p) .output (by decide) (by decide)
    i.1.1.val i.1.2.1.val i.1.2.2.1.val i.1.2.2.2.val
    i.1.1.isLt i.1.2.1.isLt i.1.2.2.1.isLt i.1.2.2.2.isLt
  have hb:=binding.backed
  have hl:=binding.lower
  have hj:=i.2.isLt
  change logicalAddress p i.1<span (typedGeometry p) .output at ha
  unfold outputByte
  omega

theorem finish_inputs_frame (p : Config) (objects : Role→TensorDescriptor)
    (binding : OutputBinding p objects) (fresh : FreshInputs objects)
    (word : Coordinate p→BF16Word) (memory : ByteMemory) (r : Role) (hr : r≠Role.output)
    (address : Fin (2^64)) (hlo : (objects r).allocationBase≤address.val)
    (hhi : address.val<(objects r).allocationBase+(objects r).allocationBytes) :
    (finishBytes p objects word memory).memory address=memory address := by
  apply finish_frame
  intro i
  have ho:=output_within_allocation p objects binding i
  have hf:=fresh r hr
  omega

#axiomsClean join_split
#axiomsClean caller_inputs
#axiomsClean outputByte_injective
#axiomsClean outputByte_bounded
#axiomsClean outputByte_descriptor
#axiomsClean byte_source_store
#axiomsClean finish_byte
#axiomsClean finish_read
#axiomsClean finish_frame
#axiomsClean output_within_allocation
#axiomsClean finish_inputs_frame
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Checked
