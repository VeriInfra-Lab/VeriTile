import bench.tritonbench_g.context_attn_llama.CheckedBridge

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnLlama Host Program
noncomputable section
set_option maxHeartbeats 4000000

abbrev Byte := BitVec 8
abbrev Word := BitVec 32
abbrev ByteMemory := Nat → Byte

def load32 (heap : ByteMemory) (p : Nat) : Word :=
  BitVec.ofNat 32 ((heap p).toNat+256*(heap (p+1)).toNat+
    65536*(heap (p+2)).toNat+16777216*(heap (p+3)).toNat)

/-- Actual FP32 Tensor plus ordinary stride/conjugate/untyped-storage queries. -/
structure Payload where
  tensor : VeriTile.Torch.FP32.Tensor
  strides : List Nat
  storageBytes : Nat
  conjugate : Bool

structure PayloadLaws (x : Payload) : Prop where
  tensor : VeriTile.Torch.FP32.TensorLaws x.tensor
  backingElements : x.storageBytes/4 = x.tensor.storage.elements
  backingSpan : x.tensor.storage.base+x.storageBytes≤2^64

def Payload.view (x : Payload) : Checked.View :=
  ⟨x.tensor.metadata,x.strides,x.tensor.storage.base,x.storageBytes,x.tensor.storageOffset,x.conjugate⟩

/-- Ordinary int32 Tensor binding for the four metadata vectors and page table. -/
structure IndexTensor where
  object : Nat
  metadata : VeriTile.Torch.TensorMetadata
  strides : List Nat
  (storageBase storageBytes storageOffset : Nat)
  physicalIndex : Nat → Nat
  conjugate : Bool

structure IndexLaws (x : IndexTensor) : Prop where
  dtype : x.metadata.dtype = .other "int32"
  layout : x.metadata.layout = .strided
  inBounds : ∀ i,i < x.metadata.numel → 4*x.physicalIndex i+4≤x.storageBytes
  contiguous : x.metadata.isContiguous=Bool.true →
    ∀ i,i < x.metadata.numel → x.physicalIndex i=x.storageOffset+i
  pointer : 0 < x.metadata.numel →
    x.metadata.dataPtr = BitVec.ofNat 64 (x.storageBase+4*x.storageOffset)
  span : x.storageBase+x.storageBytes≤2^64

def IndexTensor.view (x : IndexTensor) : Checked.View :=
  ⟨x.metadata,x.strides,x.storageBase,x.storageBytes,x.storageOffset,x.conjugate⟩

def regions : Regions := ⟨"Q","K","V","Out","Start","Sequence","Prompt","Request","Pages"⟩

structure Request where
  rawCall : Call
  dimensions : Checked.Dimensions
  (q k v out : Payload)
  (start sequence prompt request pages : IndexTensor)
  requestedMax : Nat

/-- Nominal source roles are fixed independently of physical read aliases. -/
def Request.call (r : Request) : Call := {r.rawCall with regions := regions}

inductive IndexRole where
  | start | sequence | prompt | request | pages

def IndexRole.tensor (r : Request) : IndexRole → IndexTensor
  | .start => r.start | .sequence => r.sequence | .prompt => r.prompt
  | .request => r.request | .pages => r.pages

def IndexRole.region : IndexRole → RegionName
  | .start => regions.start | .sequence => regions.sequence | .prompt => regions.prompt
  | .request => regions.request | .pages => regions.pages

def views (r : Request) : Checked.Views :=
  ⟨r.q.view,r.k.view,r.v.view,r.out.view,r.start.view,r.sequence.view,r.prompt.view,r.request.view,r.pages.view⟩

def queryCount (r : Request) : Nat := r.dimensions.queryRows*r.call.layout.heads*r.call.layout.D
def cacheCount (r : Request) : Nat := r.dimensions.cacheRows*r.call.kvHeads*r.call.layout.D
def pageCount (r : Request) : Nat := r.dimensions.requests*r.dimensions.tokens

def extent (r : Request) (rg : RegionName) : Nat :=
  if rg∈[regions.query,regions.out] then queryCount r
  else if rg∈[regions.key,regions.value] then cacheCount r
  else if rg∈[regions.start,regions.sequence,regions.prompt,regions.request] then r.call.batches
  else if rg=regions.pages then pageCount r else 0

def base (r : Request) (rg : RegionName) : Nat :=
  if rg=regions.query then r.q.tensor.dataPtr.toNat
  else if rg=regions.key then r.k.tensor.dataPtr.toNat
  else if rg=regions.value then r.v.tensor.dataPtr.toNat
  else if rg=regions.out then r.out.tensor.dataPtr.toNat
  else if rg=regions.start then r.start.metadata.dataPtr.toNat
  else if rg=regions.sequence then r.sequence.metadata.dataPtr.toNat
  else if rg=regions.prompt then r.prompt.metadata.dataPtr.toNat
  else if rg=regions.request then r.request.metadata.dataPtr.toNat
  else if rg=regions.pages then r.pages.metadata.dataPtr.toNat else 0

def indexValues (r : Request) (heap : ByteMemory) (role : IndexRole) (i : Nat) : Int :=
  (load32 heap ((role.tensor r).metadata.dataPtr.toNat+4*i)).toInt

/-- Only actual API bindings and unchanged guards over one original heap. -/
structure Contract (r : Request) (heap : ByteMemory) : Prop where
  qAPI : PayloadLaws r.q
  kAPI : PayloadLaws r.k
  vAPI : PayloadLaws r.v
  outAPI : PayloadLaws r.out
  indexAPI : ∀ role : IndexRole,IndexLaws (role.tensor r)
  checks : Checked.InputContract r.call r.dimensions (views r) (extent r) r.requestedMax
    (indexValues r heap .start) (indexValues r heap .sequence) (indexValues r heap .prompt)
    (indexValues r heap .request) (indexValues r heap .pages)

def logical {α : Type} (read : Nat → α) (x : Payload) (i : Nat) : α :=
  read (x.tensor.storage.address (x.tensor.physicalIndex i))

theorem payload_pointer (x : Payload) (laws : PayloadLaws x)
    (hc : x.tensor.metadata.isContiguous=Bool.true) (i : Nat)
    (hi : i < x.tensor.metadata.numel) :
    x.tensor.storage.address (x.tensor.physicalIndex i)=x.tensor.dataPtr.toNat+4*i := by
  have cap := VeriTile.Torch.FP32.contiguous_capacity x.tensor laws.tensor hc (by omega)
  have bound : x.tensor.storage.address x.tensor.storageOffset < 2^64 := by
    have span := x.tensor.storage.spanFits
    unfold VeriTile.Torch.FP32.Storage.address
    omega
  rw [laws.tensor.contiguous hc i hi,laws.tensor.pointer (by omega)]
  simp only [VeriTile.Torch.FP32.Storage.address] at bound
  simp only [BitVec.toNat_ofNat,VeriTile.Torch.FP32.Storage.address,Nat.mod_eq_of_lt bound]
  omega

theorem logical_direct {α : Type} (read : Nat → α) (x : Payload) (laws : PayloadLaws x)
    (hc : x.tensor.metadata.isContiguous=Bool.true) (i : Nat) (hi : i < x.tensor.metadata.numel) :
    logical read x i=read (x.tensor.dataPtr.toNat+4*i) := by
  rw [logical,payload_pointer x laws hc i hi]

theorem logical_fp32 (read : Nat → VeriTile.Torch.FP32.Word) (x : Payload) (laws : PayloadLaws x)
    (hn : x.tensor.metadata.isNeg=Bool.false) (i : Nat) :
    logical read x i=VeriTile.Torch.FP32.logicalValue read x.tensor laws.tensor i := by
  simp [logical,VeriTile.Torch.FP32.logicalValue,hn]

theorem payload_span (x : Payload) (laws : PayloadLaws x)
    (hc : x.tensor.metadata.isContiguous=Bool.true) (i : Nat)
    (hi : i < x.tensor.metadata.numel) (j : Fin 4) :
    x.tensor.storage.base≤x.tensor.dataPtr.toNat+4*i+j.val ∧
      x.tensor.dataPtr.toNat+4*i+j.val<x.tensor.storage.base+x.storageBytes := by
  rw [←payload_pointer x laws hc i hi]
  have bound := laws.tensor.inBounds i hi
  have bytes := Nat.mod_lt x.storageBytes (by decide : 0 < 4)
  have quotient := Nat.mod_add_div x.storageBytes 4
  rw [←laws.backingElements] at bound
  unfold VeriTile.Torch.FP32.Storage.address
  have := j.isLt
  omega

def metadataState (r : Request) (heap : ByteMemory) : BlockState where
  mem := fun rg off => if rg=regions.start then ⟨.int,indexValues r heap .start off⟩
    else if rg=regions.sequence then ⟨.int,indexValues r heap .sequence off⟩
    else if rg=regions.prompt then ⟨.int,indexValues r heap .prompt off⟩
    else if rg=regions.request then ⟨.int,indexValues r heap .request off⟩
    else if rg=regions.pages then ⟨.int,indexValues r heap .pages off⟩ else .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := r.call.grid.toNumPids

def metadata_observation (r : Request) (heap : ByteMemory) :
    Checked.MetadataObservation r.call (metadataState r heap)
      (indexValues r heap .start) (indexValues r heap .sequence) (indexValues r heap .prompt)
      (indexValues r heap .request) (indexValues r heap .pages) where
  startWords := fun b => load32 heap (r.start.metadata.dataPtr.toNat+4*b)
  sequenceWords := fun b => load32 heap (r.sequence.metadata.dataPtr.toNat+4*b)
  promptWords := fun b => load32 heap (r.prompt.metadata.dataPtr.toNat+4*b)
  requestWords := fun b => load32 heap (r.request.metadata.dataPtr.toNat+4*b)
  pageWords := fun b => load32 heap (r.pages.metadata.dataPtr.toNat+4*b)
  startCPU := fun _ _ => rfl
  sequenceCPU := fun _ _ => rfl
  promptCPU := fun _ _ => rfl
  requestCPU := fun _ _ => rfl
  pagesCPU := fun _ => rfl
  startMemory := by intro b hb; rfl
  sequenceMemory := by intro b hb; rfl
  promptMemory := by intro b hb; rfl
  requestMemory := by intro b hb; rfl
  pagesMemory := by intro b; rfl

theorem labels (r : Request) : Checked.Labels r.call := by
  simp [Checked.Labels,Host.Call.inputs,Request.call,regions]

#specNonCircular Contract avoiding [FP.Structural.exec,FP.Structural.run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
