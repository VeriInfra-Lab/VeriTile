import bench.tritonbench_g.attention_llama.Proofs.Tensor
import bench.tritonbench_g.attention_llama.Proofs.NativeBinding

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Backing
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionLlama HostGrid MaskCoverage
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
variable {Cell I P : Type}

/-- Exactly the two-dimensional physical launch grid. -/
abbrev Program (B NH H : Nat) := Fin (count H) × Fin (B*NH)

def owner (i : Index B NH H D) : Program B NH H :=
  ⟨(eventFor i).pid,(eventFor i).headId⟩

def anchor (hD : 0<D) (p : Program B NH H) : Index B NH H D :=
  ⟨⟨p.2.val/NH,Nat.div_lt_of_lt_mul (by simpa [Nat.mul_comm] using p.2.isLt)⟩,
    ⟨p.2.val%NH,Nat.mod_lt _ (by have h:=p.2.isLt; nlinarith)⟩,
    ⟨p.1.val*64,by have h:=p.1.isLt; simp only [count] at h; omega⟩,⟨0,hD⟩⟩

theorem owner_anchor (hD : 0<D) (p : Program B NH H) : owner (anchor hD p)=p := by
  apply Prod.ext <;> apply Fin.ext
  · simp [owner,anchor,eventFor]
  · simp only [owner,anchor,eventFor]
    simpa [Nat.mul_comm] using Nat.div_add_mod p.2.val NH

theorem initial_owner (c : Source.Config) (s : FP.Structural.State (Source.Term α))
    (hD : 0<c.D) (i : Index B c.NH c.H c.D) :
    Tensor.initialFor c s (anchor hD (owner i))=Tensor.initialFor c s i := by
  have h:=owner_anchor hD (owner i)
  have h0:=congrArg (fun p=>p.1.val) h
  have h1:=congrArg (fun p=>p.2.val) h
  change (anchor hD (owner i)).query.val/64=i.query.val/64 at h0
  change (anchor hD (owner i)).batch.val*c.NH+(anchor hD (owner i)).head.val=
    i.batch.val*c.NH+i.head.val at h1
  simp only [Tensor.initialFor,h0,h1]

def laneIndex (c : Source.Config) (hD : 0<c.D) (p : Program B c.NH c.H)
    (j : TileIndex [64,c.D])
    (hj : p.1.val*64+j.1.val<c.H) : Index B c.NH c.H c.D :=
  ⟨(anchor hD p).batch,(anchor hD p).head,⟨p.1.val*64+j.1.val,hj⟩,j.2.1⟩

theorem owner_lane (c : Source.Config) (hD : 0<c.D) (p : Program B c.NH c.H)
    (j : TileIndex [64,c.D]) (hj : p.1.val*64+j.1.val<c.H) :
    owner (laneIndex c hD p j hj)=p := by
  have h:=owner_anchor hD p
  apply Prod.ext
  · apply Fin.ext
    simp only [owner,eventFor,laneIndex]
    have hr:=j.1.isLt
    omega
  · simpa only [owner,eventFor,laneIndex] using congrArg Prod.snd h

theorem lane_address (c : Source.Config) (s : FP.Structural.State (Source.Term α))
    (hD : 0<c.D) (p : Program B c.NH c.H) (j : TileIndex [64,c.D])
    (hj : p.1.val*64+j.1.val<c.H) :
    Source.address c (Tensor.initialFor c s (anchor hD p)) j=
      Tensor.offset c (laneIndex c hD p j hj) := by
  have h0:=congrArg (fun p=>p.1.val) (owner_anchor hD p)
  simp only [owner,eventFor] at h0
  rw [Source.address,Tensor.base_eq]
  simp only [Source.rows,Tensor.initialFor,h0,↓reduceIte,Tensor.offset,laneIndex]

/-- A typed-cell heap shared by all caller Tensor objects. Addresses are
the actual byte addresses of scalar cells; scalar byte codecs remain primitive. -/
abbrev Heap (Cell : Type) := Nat→Cell

structure Pointer where
  base : Nat
  width : Nat

def Pointer.address (p : Pointer) (offset : Nat) : Nat := p.base+p.width*offset

abbrev TensorObject := Checked.TensorDescriptor

def elementOffset (t : TensorObject) (z h n d : Nat) : Nat :=
  z*t.strides[0]?.getD 0+h*t.strides[1]?.getD 0+
    n*t.strides[2]?.getD 0+d*t.strides[3]?.getD 0

def elementWidth (t : TensorObject) : Nat := if t.dtype=.int8 then 1 else 2

def tensorAddress (t : TensorObject) (i : Index B NH H D) : Nat :=
  t.dataPointer+elementWidth t*elementOffset t i.batch.val i.head.val i.query.val i.channel.val

/-- Ordinary Tensor observation: object metadata computes an address in the
supplied shared heap. There is no read callback and no expected-value field. -/
def tensorCell (heap : Heap Cell) (t : TensorObject) (i : Index B NH H D) : Cell :=
  heap (tensorAddress t i)

structure TensorLayout (t : TensorObject) (p : Pointer) (B NH H D : Nat)
    (strides : List Nat) (dtype : Checked.DType) : Prop where
  shape : t.shape=[B,NH,H,D]
  stride : t.strides=strides
  kind : t.dtype=dtype
  pointer : t.dataPointer=p.base
  width : elementWidth t=p.width
  lower : t.allocationBase≤t.dataPointer
  backed : ∀ i : Index B NH H D,
    tensorAddress t i+p.width≤t.allocationBase+t.allocationBytes
  addressFits : t.allocationBase+t.allocationBytes≤2^64

def Fresh (inputs : List TensorObject) (out : TensorObject) : Prop :=
  ∀ t∈inputs, t.allocationBase+t.allocationBytes≤out.allocationBase ∨
    out.allocationBase+out.allocationBytes≤t.allocationBase

/-- The parallel grid joins the complete disjoint source-store footprints.
Every committed value is a final typed cell of its one owning program. -/
def merge (before : Heap Cell) (address : I→Nat) (program : I→P)
    (sourceCell : P→I→Cell) : Heap Cell := by
  classical
  exact fun location=>if h : ∃ i, location=address i then
    sourceCell (program h.choose) h.choose else before location

theorem merge_read (before : Heap Cell) (address : I→Nat) (program : I→P)
    (sourceCell : P→I→Cell) (hinj : Function.Injective address) (i : I) :
    merge before address program sourceCell (address i)=sourceCell (program i) i := by
  classical
  unfold merge
  split
  · rename_i h
    have he : h.choose=i := hinj h.choose_spec.symm
    exact congrArg (fun j=>sourceCell (program j) j) he
  · rename_i h
    exact (h ⟨i,rfl⟩).elim

theorem merge_frame (before : Heap Cell) (address : I→Nat) (program : I→P)
    (sourceCell : P→I→Cell) (location : Nat) (h : ∀ i,location≠address i) :
    merge before address program sourceCell location=before location := by
  classical
  unfold merge
  apply dif_neg
  simpa only [not_exists] using h

theorem tensor_input_frame (before : Heap Cell) (address : I→Nat) (program : I→P)
    (sourceCell : P→I→Cell) (out input : TensorObject)
    (hfresh : input.allocationBase+input.allocationBytes≤out.allocationBase ∨
      out.allocationBase+out.allocationBytes ≤ input.allocationBase)
    (hout : ∀ i,out.allocationBase≤address i ∧ address i<out.allocationBase+out.allocationBytes)
    (j : Index B NH H D)
    (hj : input.allocationBase≤tensorAddress input j ∧
      tensorAddress input j < input.allocationBase+input.allocationBytes) :
    tensorCell (merge before address program sourceCell) input j=tensorCell before input j := by
  apply merge_frame
  intro i
  have hi:=hout i
  omega

#axiomsClean owner_anchor
#axiomsClean initial_owner
#axiomsClean owner_lane
#axiomsClean lane_address
#axiomsClean merge_read
#axiomsClean merge_frame
#axiomsClean tensor_input_frame
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Backing
