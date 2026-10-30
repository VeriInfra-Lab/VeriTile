import bench.repairs.tritonbench_g.streamk_matmul.Proofs.Contract
import bench.repairs.tritonbench_g.streamk_matmul.Whole001.SourceDefinitions

namespace VeriTile.Bench.Repairs.StreamK.Checked.TensorBinding
open VeriTile.Triton FP.Structural VeriTile.Torch
open Contract Whole001
noncomputable section

abbrev Word := FP.Value .fp32
abbrev ValueRead (α : Type) := Nat → α
abbrev ByteRead := Nat → BitVec 8
def readWord (read : ByteRead) (address : Nat) : Word := ⟨BitVec.ofNat 32
  ((read address).toNat+256*(read (address+1)).toNat+
    65536*(read (address+2)).toNat+16777216*(read (address+3)).toNat)⟩

structure Tensor where
  object : Nat
  metadata : TensorMetadata
  storageId : Nat
  storageBase : Nat
  storageElements : Nat
  storageOffset : Nat
  physicalIndex : Nat → Nat

/-- Ordinary FP32 Tensor storage/view laws. No kernel/source envelope occurs. -/
structure Laws (t : Tensor) : Prop where
  dtype : t.metadata.dtype=.floating .fp32
  layout : t.metadata.layout=.strided
  bounds : ∀ i,i<t.metadata.numel → t.physicalIndex i<t.storageElements
  pointer : 0<t.metadata.numel → t.metadata.dataPtr.toNat=t.storageBase+4*t.storageOffset
  contiguous : t.metadata.isContiguous=Bool.true →
    ∀ i,i<t.metadata.numel → t.physicalIndex i=t.storageOffset+i

def physicalAddress (t : Tensor) (i : Nat) := t.storageBase+4*t.physicalIndex i
def logicalValue (read : ValueRead α) (t : Tensor) (i : Nat) := read (physicalAddress t i)

theorem direct_read (read : ValueRead α) (t : Tensor) (laws : Laws t)
    (contiguous : t.metadata.isContiguous=Bool.true) (i : Nat) (hi : i<t.metadata.numel) :
    read (t.metadata.dataPtr.toNat+4*i)=logicalValue read t i := by
  rw [laws.pointer (by omega)]
  simp only [logicalValue,physicalAddress,laws.contiguous contiguous i hi,Nat.mul_add,Nat.add_assoc]

def parameters (a b : Tensor) (c : Config) (h : Profile c) : Source.Parameters where
  h := plan a.metadata b.metadata c
  strideAM := columns a.metadata
  strideAK := 1
  strideBK := columns b.metadata
  strideBN := 1
  blockM := h.1.1
  blockN := h.2.1.1
  blockK := h.2.2.1.1
  nativeK := h.2.2.2.2.1
  inputA := ⟨fun _ _ => rfl,fun _ _ => rfl⟩
  inputB := ⟨fun _ _ => rfl,fun _ _ => rfl⟩

structure Inputs (read : ValueRead α) (a b : Tensor) (c : Config) (h : Profile c)
    (s : Source.State α (parameters a b c h) exterior) : Prop where
  left : ∀ i,s.inputA i=read (a.metadata.dataPtr.toNat+4*i.val)
  right : ∀ i,s.inputB i=read (b.metadata.dataPtr.toNat+4*i.val)

def left (read : ValueRead α) (a b : Tensor) (c : Config) (h : Profile c) :
    Source.A (parameters a b c h) → α := fun i => logicalValue read a i.val
def right (read : ValueRead α) (a b : Tensor) (c : Config) (h : Profile c) :
    Source.B (parameters a b c h) → α := fun i => logicalValue read b i.val

theorem source_inputs (read : ValueRead α) (a b : Tensor) (c : Config)
    (la : Laws a) (lb : Laws b) (h : InputContract a.metadata b.metadata c)
    (s : Source.State α (parameters a b c h.profile) exterior) (bound : Inputs read a b c h.profile s) :
    s.inputA=left read a b c h.profile ∧ s.inputB=right read a b c h.profile := by
  rcases h.inputs with ⟨_,_,rankA,rankB,_,_,_,_,_,contiguousA,contiguousB,_,_,inner⟩
  constructor
  · funext i
    have hi : i.val<a.metadata.numel := by
      rw [matrix_numel a.metadata rankA]
      exact i.isLt
    exact (bound.left i).trans (direct_read read a la contiguousA i.val hi)
  · funext i
    have hi : i.val<b.metadata.numel := by
      rw [matrix_numel b.metadata rankB,←inner]
      exact i.isLt
    exact (bound.right i).trans (direct_read read b lb contiguousB i.val hi)

#axiomsClean direct_read
#axiomsClean source_inputs
end
end VeriTile.Bench.Repairs.StreamK.Checked.TensorBinding
