import VeriTile.Torch.FP32
import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Contract
import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.Grid

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001
noncomputable section

structure Payload where
  tensor : VeriTile.Torch.FP32.Tensor
  laws : VeriTile.Torch.FP32.TensorLaws tensor
  conjugate : Bool

structure Tensors where
  q : Payload
  k : Payload
  v : Payload
  history : Payload
  output : Payload

def Payload.observed (p : Payload) : Host.Tensor :=
  ⟨p.tensor.metadata.shape,p.tensor.metadata.device.index,
   p.tensor.metadata.isCuda,p.tensor.metadata.isFP32,p.tensor.metadata.isContiguous,
   p.conjugate,p.tensor.metadata.isNeg,p.tensor.dataPtr.toNat/4⟩
def Tensors.arguments (t : Tensors) (storeFlag ifcond : Bool) : Host.Arguments :=
  ⟨t.q.observed,t.k.observed,t.v.observed,some storeFlag,some ifcond⟩

/-- Concrete ABI observations of the five actual launch tensors. This carries
only ordinary pointer/shape/storage facts, never a host execution receipt,
allocator success proposition, graph equality, or a post-state condition.
The existing ABI records fresh O/H separation and permits Q/K/V aliases. -/
structure TensorBinding (c : Shape) (abi : ABI c) (t : Tensors) : Prop where
  qBase : abi.q.base=t.q.tensor.dataPtr.toNat/4
  kBase : abi.k.base=t.k.tensor.dataPtr.toNat/4
  vBase : abi.v.base=t.v.tensor.dataPtr.toNat/4
  hBase : abi.h.base=t.history.tensor.dataPtr.toNat/4
  oBase : abi.o.base=t.output.tensor.dataPtr.toNat/4
  hDevice : t.history.tensor.metadata.device=t.q.tensor.metadata.device
  oDevice : t.output.tensor.metadata.device=t.q.tensor.metadata.device
  hShape : t.history.tensor.metadata.shape=Host.historyShape c
  oShape : t.output.tensor.metadata.shape=Host.mainShape c
  hContiguous : t.history.tensor.metadata.isContiguous=Bool.true
  oContiguous : t.output.tensor.metadata.isContiguous=Bool.true
  hOrdinary : t.history.tensor.metadata.isNeg=Bool.false
  oOrdinary : t.output.tensor.metadata.isNeg=Bool.false
  hConjugate : t.history.conjugate=Bool.false
  oConjugate : t.output.conjugate=Bool.false

def TensorBinding.pointers {c : Shape} {abi : ABI c} {t : Tensors}
    (b : TensorBinding c abi t) : AttentionFwdTriton1.Checked.Pointers abi :=
  ⟨t.q.tensor.dataPtr.toNat,t.k.tensor.dataPtr.toNat,t.v.tensor.dataPtr.toNat,
   t.history.tensor.dataPtr.toNat,t.output.tensor.dataPtr.toNat,
   b.qBase,b.kBase,b.vBase,b.hBase,b.oBase⟩

abbrev Cells (A : Type) := Nat → Cell A
abbrev Word := BitVec 32

def logical {A} [Inhabited A] (before : Cells A) (t : Payload) (i : Nat) : A :=
  (before (t.tensor.storage.address (t.tensor.physicalIndex i))).read .fp32

theorem payload_address (t : Payload) (contiguous : t.tensor.metadata.isContiguous=Bool.true)
    (i : Nat) (hi : i<t.tensor.metadata.numel) :
    t.tensor.dataPtr.toNat+4*i=t.tensor.storage.address (t.tensor.physicalIndex i) := by
  have positive : 0<t.tensor.metadata.numel := by omega
  have bound := t.laws.inBounds 0 positive
  rw [t.laws.contiguous contiguous 0 positive] at bound
  have span := t.tensor.storage.spanFits
  have base : t.tensor.storage.address t.tensor.storageOffset<2^64 := by
    simp only [VeriTile.Torch.FP32.Storage.address] at *
    omega
  rw [t.laws.pointer positive]
  change (t.tensor.storage.address t.tensor.storageOffset)%2^64+4*i=_
  rw [Nat.mod_eq_of_lt base,t.laws.contiguous contiguous i hi]
  simp [VeriTile.Torch.FP32.Storage.address,Nat.mul_add,Nat.add_assoc]

theorem logical_fp32 (words : Nat → Word) (t : Payload)
    (ordinary : t.tensor.metadata.isNeg=Bool.false) (i : Nat) :
    logical (fun address => Cell.mk .fp32 (words address)) t i=
      (VeriTile.Torch.FP32.logicalValue (fun address => ⟨words address⟩) t.tensor t.laws i).bits := by
  simp [logical,VeriTile.Torch.FP32.logicalValue,ordinary]

def initial {A} [Inhabited A] (c : Shape) (abi : ABI c) (before : Cells A)
    (rest : State (Term A)) : State (Term A) :=
  {rest with mem := fun r off =>
    if r="q" then .mk .real (.input ((before (4*(abi.q.base+off))).read .fp32))
    else if r="k" then .mk .real (.input ((before (4*(abi.k.base+off))).read .fp32))
    else if r="v" then .mk .real (.input ((before (4*(abi.v.base+off))).read .fp32))
    else if r="h" then .mk .real (.input ((before (4*(abi.h.base+off))).read .fp32))
    else if r="o" then .mk .real (.input ((before (4*(abi.o.base+off))).read .fp32))
    else rest.mem r off}

def tensorInputs {A} [Inhabited A] (c : Shape) (abi : ABI c) (before : Cells A) : Inputs A :=
  ⟨fun h r j => (before (4*(abi.q.base+c.mainOffset h r j))).read .fp32,
   fun h r j => (before (4*(abi.k.base+c.mainOffset h r j))).read .fp32,
   fun h r j => (before (4*(abi.v.base+c.mainOffset h r j))).read .fp32⟩

#axiomsClean payload_address
#axiomsClean logical_fp32
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked
