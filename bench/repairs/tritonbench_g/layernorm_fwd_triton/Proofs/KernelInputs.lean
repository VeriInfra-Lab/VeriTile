import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.Address
import bench.repairs.tritonbench_g.layernorm_fwd_triton.Contract
import VeriTile.Torch.FP32
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.InputBinding
open VeriTile.Triton FP.Structural Address
namespace Torch
export VeriTile.Torch.FP32 (Tensor TensorLaws Word Storage logicalValue)
end Torch
noncomputable section

variable {shape : Shape}

def index3 (row : Row shape) (col : Nat) : Nat :=
  (row.1.val*shape.heads+row.2.val)*shape.columns+col
def index2 (row : Row shape) (col : Nat) : Nat := row.2.val*shape.columns+col

/-- Original Tensor API interpretation of the actual three stride arguments.
It imposes no contiguity, positivity of a dimension/stride, or offset-zero rule. -/
structure View3 (t : Torch.Tensor) (shape : Shape) (seqStride headStride dimStride : Nat) : Prop where
  dimensions : t.metadata.shape=[shape.sequences,shape.heads,shape.columns]
  index : ∀ row : Row shape, ∀ col : Fin shape.columns,
    t.physicalIndex (index3 row col.val)=t.storageOffset+
      offset row.1.val row.2.val col.val seqStride headStride dimStride
  shapeSigned : shape.sequences<2^63 ∧ shape.heads<2^63 ∧ shape.columns<2^63
  signed : seqStride<2^63 ∧ headStride<2^63 ∧ dimStride<2^63

structure View2 (t : Torch.Tensor) (shape : Shape) (headStride dimStride : Nat) : Prop where
  dimensions : t.metadata.shape=[shape.heads,shape.columns]
  index : ∀ row : Row shape, ∀ col : Fin shape.columns,
    t.physicalIndex (index2 row col.val)=t.storageOffset+
      offset 0 row.2.val col.val 0 headStride dimStride
  shapeSigned : shape.heads<2^63 ∧ shape.columns<2^63
  signed : headStride<2^63 ∧ dimStride<2^63

theorem index2_lt (row : Row shape) (col : Fin shape.columns) :
    index2 row col.val<shape.heads*shape.columns := by
  have h := Nat.mul_le_mul_right shape.columns (Nat.succ_le_of_lt row.2.isLt)
  unfold index2
  rw [Nat.succ_mul] at h
  omega

theorem index3_lt (row : Row shape) (col : Fin shape.columns) :
    index3 row col.val<shape.sequences*(shape.heads*shape.columns) := by
  have h := Nat.mul_le_mul_right (shape.heads*shape.columns) (Nat.succ_le_of_lt row.1.isLt)
  have hc := index2_lt row col
  unfold index3 index2 at *
  rw [Nat.succ_mul] at h
  rw [Nat.add_mul,Nat.mul_assoc]
  omega

def address (t : Torch.Tensor) (offset : Nat) : Nat := t.dataPtr.toNat+4*offset

/-- Active pointer arithmetic equals the original storage coordinate. Bounds
come from the Tensor laws; no raw-word or output equality is assumed. -/
theorem address_logical (read : Nat → Torch.Word) (t : Torch.Tensor) (laws : Torch.TensorLaws t)
    (notNeg : t.metadata.isNeg=Bool.false) (i off : Nat) (hi : i<t.metadata.numel)
    (index : t.physicalIndex i=t.storageOffset+off) :
    address t off=t.storage.address (t.physicalIndex i) ∧ address t off+4≤2^64 ∧
    (t.dataPtr+BitVec.ofNat 64 (4*off)).toNat=address t off ∧
    read (address t off)=Torch.logicalValue read t laws i := by
  have hb := laws.inBounds i hi
  have span := t.storage.spanFits
  have ha : t.storage.address (t.physicalIndex i)+4≤2^64 := by
    simp only [VeriTile.Torch.FP32.Storage.address]
    omega
  have base : t.storage.address t.storageOffset<2^64 := by
    simp only [VeriTile.Torch.FP32.Storage.address,index] at ha ⊢
    omega
  have hp := congrArg BitVec.toNat (laws.pointer (by omega))
  simp only [BitVec.toNat_ofNat,Nat.mod_eq_of_lt base] at hp
  have he : address t off=t.storage.address (t.physicalIndex i) := by
    simp only [address,hp,VeriTile.Torch.FP32.Storage.address,index]
    omega
  have hab : address t off+4≤2^64 := he ▸ ha
  have pn : (t.dataPtr+BitVec.ofNat 64 (4*off)).toNat=address t off := by
    rw [BitVec.toNat_add,BitVec.toNat_ofNat,Nat.add_mod_mod]
    exact Nat.mod_eq_of_lt (by unfold address at hab; omega)
  exact ⟨he,hab,pn,by simp [he,VeriTile.Torch.FP32.logicalValue,notNeg]⟩

/-- The actual kernel arguments and their original backing storages. Output
injectivity/freshness describe the supplied writable buffer, not host code. -/
structure Arguments (shape : Shape) (st : Strides) (eps : Contract.Epsilon) where
  (x w y : Torch.Tensor)
  (xLaws : Torch.TensorLaws x) (wLaws : Torch.TensorLaws w) (yLaws : Torch.TensorLaws y)
  xView : View3 x shape st.xSeq st.xHead st.xDim
  wView : View2 w shape st.wHead st.wDim
  yView : View3 y shape st.ySeq st.yHead st.yDim
  admitted : Contract.InputContract x.metadata w.metadata eps
  yNoNeg : y.metadata.isNeg=Bool.false
  outputInjective : Function.Injective (fun q : Coordinate shape => yOffset st q.1 q.2.val)
  yAligned : 0<y.metadata.numel → y.dataPtr.toNat%4=0
  yDevice : y.metadata.device=x.metadata.device
  fresh : ∀ t, t=x ∨ t=w → ∀ i, i<t.storage.elements → ∀ row : Row shape,
    ∀ col : Fin shape.columns, ∀ b, b<4 →
      t.storage.address i+b<address y (yOffset st row col.val) ∨
        address y (yOffset st row col.val)+4≤t.storage.address i+b

variable {st : Strides} {eps : Contract.Epsilon}

theorem x_pointer (a : Arguments shape st eps) (read : Nat → Torch.Word)
    (row : Row shape) (col : Fin shape.columns) :
    address a.x (xOffset st row col.val)=a.x.storage.address (a.x.physicalIndex (index3 row col.val)) ∧
    address a.x (xOffset st row col.val)+4≤2^64 ∧
    (a.x.dataPtr+BitVec.ofNat 64 (4*xOffset st row col.val)).toNat=address a.x (xOffset st row col.val) ∧
    read (address a.x (xOffset st row col.val))=Torch.logicalValue read a.x a.xLaws (index3 row col.val) := by
  apply address_logical read a.x a.xLaws a.admitted.input.noNeg
  · simpa [VeriTile.Torch.TensorMetadata.numel,a.xView.dimensions] using index3_lt row col
  · exact a.xView.index row col

theorem w_pointer (a : Arguments shape st eps) (read : Nat → Torch.Word)
    (row : Row shape) (col : Fin shape.columns) :
    address a.w (wOffset st row col.val)=a.w.storage.address (a.w.physicalIndex (index2 row col.val)) ∧
    address a.w (wOffset st row col.val)+4≤2^64 ∧
    (a.w.dataPtr+BitVec.ofNat 64 (4*wOffset st row col.val)).toNat=address a.w (wOffset st row col.val) ∧
    read (address a.w (wOffset st row col.val))=Torch.logicalValue read a.w a.wLaws (index2 row col.val) := by
  apply address_logical read a.w a.wLaws a.admitted.weight.noNeg
  · simpa [VeriTile.Torch.TensorMetadata.numel,a.wView.dimensions] using index2_lt row col
  · exact a.wView.index row col

theorem y_pointer (a : Arguments shape st eps) (row : Row shape) (col : Fin shape.columns) :
    address a.y (yOffset st row col.val)=a.y.storage.address (a.y.physicalIndex (index3 row col.val)) ∧
    address a.y (yOffset st row col.val)+4≤2^64 ∧
    (a.y.dataPtr+BitVec.ofNat 64 (4*yOffset st row col.val)).toNat=address a.y (yOffset st row col.val) := by
  have h := address_logical (fun _ => FP.Value.zero .fp32) a.y a.yLaws a.yNoNeg
    (index3 row col.val) (yOffset st row col.val) (by
      simpa [VeriTile.Torch.TensorMetadata.numel,a.yView.dimensions] using index3_lt row col)
    (a.yView.index row col)
  exact ⟨h.1,h.2.1,h.2.2.1⟩

/-- Shared raw physical read ports preserve arbitrary read-only X/W aliasing. -/
def state {α : Type} (read : RegionName → Nat → α) (a : Arguments shape st eps)
    (scratch : State α) : State α :=
  {scratch with mem := fun region off =>
    if region="X" then .mk .fp32 (read "X" (address a.x off))
    else if region="W" then .mk .fp32 (read "W" (address a.w off))
    else scratch.mem region off}

def logicalX {α : Type} (read : RegionName → Nat → α) (a : Arguments shape st eps)
    (row : Row shape) (col : Nat) : α :=
  read "X" (a.x.storage.address (a.x.physicalIndex (index3 row col)))
def logicalW {α : Type} (read : RegionName → Nat → α) (a : Arguments shape st eps)
    (row : Row shape) (col : Nat) : α :=
  read "W" (a.w.storage.address (a.w.physicalIndex (index2 row col)))

#axiomsClean address_logical
#axiomsClean x_pointer
#axiomsClean w_pointer
#axiomsClean y_pointer
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.InputBinding
