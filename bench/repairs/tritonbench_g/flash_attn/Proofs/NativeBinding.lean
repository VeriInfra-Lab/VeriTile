import bench.repairs.tritonbench_g.flash_attn.Contract

namespace VeriTile.Bench.Repairs.FlashAttn.Checked
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000

abbrev ByteMemory := Fin (2^64)→Physical.Byte
def extendMemory (memory : ByteMemory) (n : Nat) : Physical.Byte :=
  if h : n<2^64 then memory ⟨n,h⟩ else 0

/-- Actual fresh contiguous FP32 scratch allocation observations. -/
structure Scratch where
  (identity allocation storageBase storageBytes dataPointer device : Nat)
  (shape strides : List Nat)
  dtypeFP32 : Bool
  pointer : dataPointer=storageBase

structure ScratchBinding (a : Host.Arguments) (scratch : Scratch) : Prop where
  shape : scratch.shape=[a.q.shape.batches*a.q.shape.heads,a.q.shape.length]
  strides : scratch.strides=[a.q.shape.length,1]
  device : scratch.device=a.q.device
  dtype : scratch.dtypeFP32=Bool.true
  backed : 4*(a.q.shape.batches*a.q.shape.heads*a.q.shape.length)≤scratch.storageBytes

def pointers (q k v o : Tensor) (scratch : Scratch) : Region→Nat
  | .q=>q.dataPointer | .k=>k.dataPointer | .v=>v.dataPointer
  | .o=>o.dataPointer | .l=>scratch.dataPointer | .other _=>0

/-- Every load in the graph reads four bytes at the actual caller pointer. -/
def callerBytes (base : Region→Nat) (memory : ByteMemory) : Physical.Bytes :=
  fun r offset=>extendMemory memory (base r+offset)

def readWord (memory : ByteMemory) (base : Nat) : Word :=
  Physical.decode (fun j=>extendMemory memory (base+j.val))

theorem caller_load (base : Region→Nat) (memory : ByteMemory) (r : Region) (offset : Nat) :
    Physical.load (callerBytes base memory) r offset=readWord memory (base r+4*offset) := by
  simp only [Physical.load,callerBytes,readWord,Nat.add_assoc]

def StoredByte (x : Layout) (base : Region→Nat) (r : Region) (address : Nat) : Prop :=
  base r≤address ∧ footprint x r ((address-base r)/4)

/-- Ordinary mapping of the executed relative source byte stores into the
actual finite address space. O and L have separate fresh allocations. -/
def finishMemory (x : Layout) (base : Region→Nat) (native : Native.Instructions) (scale : Word)
    (memory : ByteMemory) : ByteMemory := fun address=>
  let final:=Physical.byteExecute x (Native.ops native) scale (sourceSchedule x.shape) (callerBytes base memory)
  if StoredByte x base .o address.val then final .o (address.val-base .o)
  else if StoredByte x base .l address.val then final .l (address.val-base .l)
  else memory address

theorem stored_output (x : Layout) (base : Region→Nat) (b h n d : Nat)
    (active : (Work.output b h n d).active x.shape) (j : Fin 4) :
    StoredByte x base .o (base .o+4*address x.o b h n d+j.val) := by
  refine ⟨by omega,?_⟩
  have he : (base .o+4*address x.o b h n d+j.val-base .o)/4=address x.o b h n d := by have:=j.isLt;omega
  rw [he]
  exact ⟨.output b h n d,active,rfl⟩

theorem stored_scratch (x : Layout) (base : Region→Nat) (b h n : Nat)
    (active : (Work.logsum b h n).active x.shape) (j : Fin 4) :
    StoredByte x base .l (base .l+4*lAddress x.shape b h n+j.val) := by
  refine ⟨by omega,?_⟩
  have he : (base .l+4*lAddress x.shape b h n+j.val-base .l)/4=lAddress x.shape b h n := by have:=j.isLt;omega
  rw [he]
  exact ⟨.logsum b h n,active,rfl⟩

theorem output_allocation (a : Host.Arguments) (q k v o : Tensor) (scratch : Scratch)
    (allocation : Allocations a o scratch.allocation) (order : SortedAxes) (ho : tensorChecks o order=Bool.true)
    (location : Nat) (stored : StoredByte (Host.layout a o.tensor) (pointers q k v o scratch) .o location) :
    o.storageBase≤location ∧ location<o.storageBase+o.tensor.storageBytes := by
  rcases stored with ⟨hlo,w,hw,he⟩
  cases w with
  | logsum b h n=>simp [Work.target] at he
  | output b h n d=>
    have he' : (location-o.dataPointer)/4=address o.tensor.strides b h n d := by simpa [pointers,Work.target,Host.layout] using congrArg Prod.snd he
    have ha : b<o.tensor.shape.batches ∧ h<o.tensor.shape.heads ∧ n<o.tensor.shape.length ∧ d<o.tensor.shape.dim := by
      simpa only [Work.active,Host.layout,Host.shape,allocation.outputShape] using hw
    have hb:=Host.tensor_access o.tensor (admitted_tensor o order ho) b h n d ha.1 ha.2.1 ha.2.2.1 ha.2.2.2
    have hp:=o.pointer
    change o.dataPointer≤location at hlo
    omega

theorem scratch_allocation (a : Host.Arguments) (q k v o : Tensor) (scratch : Scratch)
    (binding : ScratchBinding a scratch) (location : Nat)
    (stored : StoredByte (Host.layout a o.tensor) (pointers q k v o scratch) .l location) :
    scratch.storageBase≤location ∧ location<scratch.storageBase+scratch.storageBytes := by
  rcases stored with ⟨hlo,w,hw,he⟩
  cases w with
  | output b h n d=>simp [Work.target] at he
  | logsum b h n=>
    have he' : (location-scratch.dataPointer)/4=lAddress (Host.shape a) b h n := by
      simpa [pointers,Work.target,Host.layout] using congrArg Prod.snd he
    have hb:=l_bound (Host.shape a) b h n hw.1 hw.2.1 hw.2.2
    have hs:=binding.backed
    have hp:=scratch.pointer
    change scratch.dataPointer≤location at hlo
    change lAddress (Host.shape a) b h n<a.q.shape.batches*a.q.shape.heads*a.q.shape.length at hb
    omega

/-- Ordinary allocator interval separation; no access or numerical fact. -/
structure Fresh (q k v o : Tensor) (scratch : Scratch) : Prop where
  outputScratch : o.storageBase+o.tensor.storageBytes≤scratch.storageBase ∨
    scratch.storageBase+scratch.storageBytes≤o.storageBase
  outputInputs : ∀ t∈[q,k,v],o.storageBase+o.tensor.storageBytes≤t.storageBase ∨
    t.storageBase+t.tensor.storageBytes≤o.storageBase
  scratchInputs : ∀ t∈[q,k,v],scratch.storageBase+scratch.storageBytes≤t.storageBase ∨
    t.storageBase+t.tensor.storageBytes≤scratch.storageBase

theorem output_byte (a : Host.Arguments) (q k v o : Tensor) (scratch : Scratch)
    (valid : Host.Contract a) (allocation : Allocations a o scratch.allocation)
    (order : SortedAxes) (ho : tensorChecks o order=Bool.true)
    (native : Native.Instructions) (memory : ByteMemory) (b h n d : Nat)
    (active : (Work.output b h n d).active (Host.shape a)) (j : Fin 4) :
    extendMemory (finishMemory (Host.layout a o.tensor) (pointers q k v o scratch) native a.scale memory)
      (o.dataPointer+4*address o.tensor.strides b h n d+j.val)=
      Physical.encode (outputValue (Host.layout a o.tensor) (Native.ops native)
        (Physical.load (callerBytes (pointers q k v o scratch) memory)) a.scale b h n d) j := by
  have ha : b<o.tensor.shape.batches ∧ h<o.tensor.shape.heads ∧ n<o.tensor.shape.length ∧ d<o.tensor.shape.dim := by
    simpa only [Work.active,Host.shape,allocation.outputShape] using active
  have hb:=(formed_tensor o order ho b h n d ha.1 ha.2.1 (by omega) ha.2.2.2 j).2.2
  have hs:=stored_output (Host.layout a o.tensor) (pointers q k v o scratch) b h n d active j
  change StoredByte (Host.layout a o.tensor) (pointers q k v o scratch) .o
    (o.dataPointer+4*address o.tensor.strides b h n d+j.val) at hs
  rw [extendMemory,dif_pos hb]
  simp only [finishMemory]
  rw [if_pos hs]
  have he : o.dataPointer+4*address o.tensor.strides b h n d+j.val-(pointers q k v o scratch) .o=
      4*address o.tensor.strides b h n d+j.val := by simp [pointers];omega
  rw [he]
  exact Physical.whole_bytes _ (Host.contract_shape a valid)
    (Host.output_layout a o.tensor scratch.allocation (admitted_allocations a o scratch.allocation allocation order ho))
    native native (Native.reflexive native) a.scale _ (List.Perm.refl _) _ (.output b h n d) active j

theorem scratch_byte (a : Host.Arguments) (q k v o : Tensor) (scratch : Scratch)
    (valid : Host.Contract a) (allocation : Allocations a o scratch.allocation)
    (order : SortedAxes) (ho : tensorChecks o order=Bool.true) (binding : ScratchBinding a scratch)
    (fresh : Fresh q k v o scratch) (hp : scratchPointerChecks (Host.shape a) scratch.dataPointer=Bool.true)
    (native : Native.Instructions) (memory : ByteMemory) (b h n : Nat)
    (active : (Work.logsum b h n).active (Host.shape a)) (j : Fin 4) :
    extendMemory (finishMemory (Host.layout a o.tensor) (pointers q k v o scratch) native a.scale memory)
      (scratch.dataPointer+4*lAddress (Host.shape a) b h n+j.val)=
      Physical.encode (logValue (Host.layout a o.tensor) (Native.ops native)
        (Physical.load (callerBytes (pointers q k v o scratch) memory)) a.scale b h n) j := by
  have hs:=stored_scratch (Host.layout a o.tensor) (pointers q k v o scratch) b h n active j
  change StoredByte (Host.layout a o.tensor) (pointers q k v o scratch) .l
    (scratch.dataPointer+4*lAddress (Host.shape a) b h n+j.val) at hs
  have alloc:=scratch_allocation a q k v o scratch binding _ hs
  have no : ¬StoredByte (Host.layout a o.tensor) (pointers q k v o scratch) .o
      (scratch.dataPointer+4*lAddress (Host.shape a) b h n+j.val) := by
    intro h
    have other:=output_allocation a q k v o scratch allocation order ho _ h
    have hf:=fresh.outputScratch
    omega
  have hb:=l_bound (Host.shape a) b h n active.1 active.2.1 active.2.2
  simp only [scratchPointerChecks,decide_eq_true_eq] at hp
  have bounded : scratch.dataPointer+4*lAddress (Host.shape a) b h n+j.val<2^64 := by
    have:=j.isLt
    omega
  rw [extendMemory,dif_pos bounded]
  simp only [finishMemory]
  rw [if_neg no,if_pos hs]
  have he : scratch.dataPointer+4*lAddress (Host.shape a) b h n+j.val-(pointers q k v o scratch) .l=
      4*lAddress (Host.shape a) b h n+j.val := by simp [pointers];omega
  rw [he]
  exact Physical.whole_bytes _ (Host.contract_shape a valid)
    (Host.output_layout a o.tensor scratch.allocation (admitted_allocations a o scratch.allocation allocation order ho))
    native native (Native.reflexive native) a.scale _ (List.Perm.refl _) _ (.logsum b h n) active j

theorem finish_frame (x : Layout) (base : Region→Nat) (native : Native.Instructions) (scale : Word)
    (memory : ByteMemory) (location : Fin (2^64))
    (ho : ¬StoredByte x base .o location.val) (hl : ¬StoredByte x base .l location.val) :
    finishMemory x base native scale memory location=memory location := by
  simp [finishMemory,ho,hl]

theorem input_frame (a : Host.Arguments) (q k v o : Tensor) (scratch : Scratch)
    (allocation : Allocations a o scratch.allocation) (order : SortedAxes) (ho : tensorChecks o order=Bool.true)
    (binding : ScratchBinding a scratch) (fresh : Fresh q k v o scratch)
    (native : Native.Instructions) (memory : ByteMemory) (t : Tensor) (ht : t∈[q,k,v])
    (location : Fin (2^64)) (hlo : t.storageBase≤location.val)
    (hhi : location.val<t.storageBase+t.tensor.storageBytes) :
    finishMemory (Host.layout a o.tensor) (pointers q k v o scratch) native a.scale memory location=memory location := by
  apply finish_frame
  · intro h
    have bound:=output_allocation a q k v o scratch allocation order ho _ h
    have hf:=fresh.outputInputs t ht
    omega
  · intro h
    have bound:=scratch_allocation a q k v o scratch binding _ h
    have hf:=fresh.scratchInputs t ht
    omega

structure Returned where
  tensor : Tensor
  scratch : Scratch
  memory : ByteMemory
  (launchDevice restoredDevice : Nat)

def invoke (a : Host.Arguments) (q k v o : Tensor) (scratch : Scratch) (callerDevice : Nat)
    (native : Native.Instructions) (memory : ByteMemory) : Returned :=
  ⟨o,scratch,finishMemory (Host.layout a o.tensor) (pointers q k v o scratch) native a.scale memory,
    a.q.device,callerDevice⟩

def Returned.read (result : Returned) (b h n d : Nat) : Word :=
  readWord result.memory (result.tensor.dataPointer+4*address result.tensor.tensor.strides b h n d)
def Returned.readScratch (result : Returned) (head n : Nat) : Word :=
  readWord result.memory (result.scratch.dataPointer+
    4*(head*result.scratch.strides[0]?.getD 0+n*result.scratch.strides[1]?.getD 0))

theorem returned_word (a : Host.Arguments) (q k v o : Tensor) (scratch : Scratch)
    (valid : Host.Contract a) (allocation : Allocations a o scratch.allocation)
    (order : SortedAxes) (ho : tensorChecks o order=Bool.true)
    (callerDevice : Nat) (native : Native.Instructions) (memory : ByteMemory) (b h n d : Nat)
    (active : (Work.output b h n d).active (Host.shape a)) :
    (invoke a q k v o scratch callerDevice native memory).read b h n d=
      outputValue (Host.layout a o.tensor) (Native.ops native)
        (Physical.load (callerBytes (pointers q k v o scratch) memory)) a.scale b h n d := by
  unfold Returned.read readWord
  have he:=funext (output_byte a q k v o scratch valid allocation order ho native memory b h n d active)
  change Physical.decode (fun j=>extendMemory
    (finishMemory (Host.layout a o.tensor) (pointers q k v o scratch) native a.scale memory)
    (o.dataPointer+4*address o.tensor.strides b h n d+j.val))=_
  rw [he,Physical.decode_encode]

theorem returned_scratch_word (a : Host.Arguments) (q k v o : Tensor) (scratch : Scratch)
    (valid : Host.Contract a) (allocation : Allocations a o scratch.allocation)
    (order : SortedAxes) (ho : tensorChecks o order=Bool.true) (binding : ScratchBinding a scratch)
    (fresh : Fresh q k v o scratch) (hp : scratchPointerChecks (Host.shape a) scratch.dataPointer=Bool.true)
    (callerDevice : Nat) (native : Native.Instructions) (memory : ByteMemory) (b h n : Nat)
    (active : (Work.logsum b h n).active (Host.shape a)) :
    (invoke a q k v o scratch callerDevice native memory).readScratch (b*a.q.shape.heads+h) n=
      logValue (Host.layout a o.tensor) (Native.ops native)
        (Physical.load (callerBytes (pointers q k v o scratch) memory)) a.scale b h n := by
  have he:=funext (scratch_byte a q k v o scratch valid allocation order ho binding fresh hp native memory b h n active)
  simp only [Returned.readScratch,invoke,binding.strides,List.getElem?_cons_zero,List.getElem?_cons_succ,
    Option.getD_some,Nat.mul_one,readWord]
  change Physical.decode (fun j=>extendMemory
    (finishMemory (Host.layout a o.tensor) (pointers q k v o scratch) native a.scale memory)
    (scratch.dataPointer+4*lAddress (Host.shape a) b h n+j.val))=_
  rw [he,Physical.decode_encode]

#axiomsClean caller_load
#axiomsClean stored_output
#axiomsClean stored_scratch
#axiomsClean output_allocation
#axiomsClean scratch_allocation
#axiomsClean output_byte
#axiomsClean scratch_byte
#axiomsClean finish_frame
#axiomsClean input_frame
#axiomsClean returned_word
#axiomsClean returned_scratch_word
end
end VeriTile.Bench.Repairs.FlashAttn.Checked
