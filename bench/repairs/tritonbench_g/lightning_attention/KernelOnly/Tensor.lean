import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.ByteCodec
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Model
import bench.repairs.tritonbench_g.lightning_attention.Proofs.HostMetadata
import VeriTile.Torch.FP32
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly
open VeriTile Triton
noncomputable section
set_option maxHeartbeats 4000000
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


inductive Phase where
  | forward | backward
  deriving DecidableEq

def outputs : Phase → Nat | .forward => 1 | .backward => 3
instance : NeZero (outputs .forward) := ⟨by decide⟩
instance : NeZero (outputs .backward) := ⟨by decide⟩
structure Incoming where
  original : Payload
  dense : Payload

/-- Forward carries only its actual arguments; backward has its current saved
Q/K/V, original incoming Tensor, and the ordinary contiguous preparation. -/
structure Request (phase : Phase) where
  dimensions : Host.Dimensions
  device : Torch.Device
  q : Payload
  k : Payload
  v : Payload
  result : Fin (outputs phase) → Payload
  incoming : match phase with | .forward => PUnit | .backward => Incoming

def incomingReads : {phase : Phase} → Request phase → List Payload
  | .forward,_ => []
  | .backward,r => [r.incoming.dense]
def incomingBackings : {phase : Phase} → Request phase → List Payload
  | .forward,_ => []
  | .backward,r => [r.incoming.original,r.incoming.dense]
def reads {phase} (r : Request phase) : List Payload := [r.q,r.k,r.v] ++ incomingReads r
def readBackings {phase} (r : Request phase) : List Payload := [r.q,r.k,r.v] ++ incomingBackings r

def resultWidth {phase} (r : Request phase) (j : Fin (outputs phase)) : Nat :=
  match phase with
  | .forward => r.dimensions.E
  | .backward => if j.val<2 then r.dimensions.D else r.dimensions.E

def resultRegion : {phase : Phase} → Fin (outputs phase) → RegionName
  | .forward,_ => "output"
  | .backward,j => if j.val=0 then "queryGradient" else if j.val=1 then "keyGradient" else "valueGradient"

def inputG : {phase : Phase} → Request phase → Option Payload
  | .forward,_ => none
  | .backward,r => some r.incoming.dense

def argument {phase} (r : Request phase) (rg : RegionName) : Option Payload :=
  if rg="query" then some r.q else if rg="key" then some r.k else if rg="value" then some r.v
  else if rg="incoming" then inputG r
  else (List.finRange (outputs phase)).findSome? fun j =>
    if rg=resultRegion j then some (r.result j) else none

def base {phase} (r : Request phase) (rg : RegionName) : Nat :=
  ((argument r rg).map (fun x => x.tensor.dataPtr.toNat)).getD 0

def separate (x y : Payload) : Prop :=
  x.tensor.storage.base+x.storageBytes≤y.tensor.storage.base ∨
    y.tensor.storage.base+y.storageBytes≤x.tensor.storage.base

def Descriptor (d : Host.Dimensions) (device : Torch.Device) (x : Payload) (width : Nat)
    (contiguous : Bool := Bool.true) : Prop :=
  PayloadLaws x ∧ x.tensor.metadata.shape=[d.B,d.H,d.N,width] ∧
  x.tensor.metadata.device=device ∧ (contiguous=Bool.true → x.tensor.metadata.isContiguous=Bool.true) ∧
  x.tensor.metadata.isNeg=Bool.false ∧ x.conjugate=Bool.false

def IncomingContract : {phase : Phase} → Request phase → Prop
  | .forward,_ => True
  | .backward,r => Descriptor r.dimensions r.device r.incoming.original r.dimensions.E Bool.false ∧
      Descriptor r.dimensions r.device r.incoming.dense r.dimensions.E

/-- Metadata, actual pointers and full backing separation only. This admits
arbitrary contiguous strides/offsets and every read alias, with no resource cap. -/
structure Contract {phase} (r : Request phase) : Prop where
  dimensions : Host.Contract r.dimensions
  cuda : r.device.kind="cuda"
  q : Descriptor r.dimensions r.device r.q r.dimensions.D
  k : Descriptor r.dimensions r.device r.k r.dimensions.D
  v : Descriptor r.dimensions r.device r.v r.dimensions.E
  incoming : IncomingContract r
  result : ∀ j, Descriptor r.dimensions r.device (r.result j) (resultWidth r j)
  outputRead : ∀ j,∀ x∈readBackings r,separate (r.result j) x
  outputOutput : ∀ i j,i≠j → separate (r.result i) (r.result j)
  pointer : ∀ x∈reads r ++ (List.finRange (outputs phase)).map r.result,
    x.tensor.dataPtr.toNat%4=0 ∧ x.tensor.dataPtr.toNat+4*r.dimensions.formedLimit<2^63

/-- Ordinary contiguous-input preparation law, at the same current memory.
It concerns only incoming input coordinates; no execution/output result occurs. -/
def Prepared {α : Type} (r : Request .backward) (read : Nat → α) : Prop :=
  ∀ i,i<r.incoming.original.tensor.metadata.numel →
    logical read r.incoming.dense i=logical read r.incoming.original i

def Preparation {α : Type} : {phase : Phase} → (r : Request phase) → (Nat → α) → Prop
  | .forward,_,_ => True
  | .backward,r,read => Prepared r read

theorem descriptor_numel (d : Host.Dimensions) (device : Torch.Device) (x : Payload) (w : Nat)
    (c : Bool) (h : Descriptor d device x w c) : x.tensor.metadata.numel=d.B*d.H*d.N*w := by
  simp [Torch.TensorMetadata.numel,h.2.1,Nat.mul_assoc]

#specNonCircular Contract avoiding [FP.Structural.exec,FP.Structural.run]
#specNonCircular Prepared avoiding [FP.Structural.exec,FP.Structural.run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly
