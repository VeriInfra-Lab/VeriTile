import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.NativeReturn
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.TensorBinding
open VeriTile Triton AttnFwdTriton TypedExecution Checked
open Verified.PhysicalCells (Role)
noncomputable section

/-- Flattening is only the Tensor API logical coordinate. Each role keeps its
own physical strides and may share read-only storage with any other role. -/
def logicalIndex (p : Config) (r : Role) (z h n d : Nat) : Nat :=
  match r with
  | .queryScale | .keyScale=>n
  | _=>((z*p.heads+h)*p.context+n)*p.headDim+d

/-- Ordinary strided Tensor API binding for the already allocated caller
objects. Fields concern only input metadata, identity, storage and indexing. -/
structure View (p : Config) (objects : Role→TensorDescriptor)
    (tensors : Role→Torch.FP32.Tensor) : Prop where
  laws : ∀ r,r≠Role.output → Torch.FP32.TensorLaws (tensors r)
  object : ∀ r,r≠Role.output → (objects r).object=(tensors r).object
  storage : ∀ r,r≠Role.output → (objects r).storage=(tensors r).storage.object
  pointer : ∀ r,r≠Role.output → (objects r).dataPointer=(tensors r).dataPtr.toNat
  allocationBase : ∀ r,r≠Role.output → (objects r).allocationBase=(tensors r).storage.base
  allocationBytes : ∀ r,r≠Role.output → (objects r).allocationBytes=4*(tensors r).storage.elements
  fp32 : ∀ r,r≠Role.output → (objects r).metadata.fp32=(tensors r).metadata.isFP32
  stridedLayout : ∀ r,r≠Role.output → (objects r).metadata.strided=(tensors r).metadata.isStrided
  cuda : ∀ r,r≠Role.output → (objects r).metadata.cuda=(tensors r).metadata.isCuda
  contiguous : ∀ r,r≠Role.output → (objects r).metadata.contiguous=(tensors r).metadata.isContiguous
  numel : ∀ r,r≠Role.output → (objects r).metadata.numel=(tensors r).metadata.numel
  negative : ∀ r,r≠Role.output → (objects r).metadata.negative=(tensors r).metadata.isNeg
  shape : ∀ r,r≠Role.output → (objects r).metadata.shape=(tensors r).metadata.shape
  device : ∀ r,r≠Role.output → (objects r).metadata.device=(tensors r).metadata.device.index
  strided : ∀ r,r≠Role.output → ∀ z h n d,roleActive p z h n d r →
    (tensors r).physicalIndex (logicalIndex p r z h n d)=
      (tensors r).storageOffset+(match r with
      | .queryScale | .keyScale=>n
      | _=>descriptorElement (objects r) z h n d)

private theorem flatten_lt (B H N D z h n d : Nat) (hz : z<B) (hh : h<H)
    (hn : n<N) (hd : d<D) : ((z*H+h)*N+n)*D+d < B*H*N*D := by
  have step (a limit width b : Nat) (ha : a<limit) (hb : b<width) : a*width+b<limit*width := by
    have hm:=Nat.mul_le_mul_right width (Nat.succ_le_of_lt ha)
    nlinarith
  exact step _ _ _ _ (step _ _ _ _ (step _ _ _ _ hz hh) hn) hd

/-- Logical index bounds follow from actual shape observations and the
source's existing packed-scale admission checks. No bound is a View field. -/
theorem valid_index (p : Config) (objects : Role→TensorDescriptor)
    (tensors : Role→Torch.FP32.Tensor) (view : View p objects tensors)
    (hinput : inputChecks (typedGeometry p) (objects .query).metadata (objects .key).metadata
      (objects .value).metadata (objects .queryScale).metadata (objects .keyScale).metadata=Bool.true)
    (hl : Checked.launchChecks (typedGeometry p) (objects .queryScale).metadata
      (objects .keyScale).metadata=Bool.true)
    (r : Role) (hr : r≠Role.output) (z h n d : Nat) (ha : roleActive p z h n d r) :
    logicalIndex p r z h n d < (tensors r).metadata.numel := by
  simp only [inputChecks,Bool.and_eq_true,decide_eq_true_eq] at hinput
  simp only [Checked.launchChecks,decide_eq_true_eq] at hl
  have matrix (r : Role) (hr : r≠Role.output)
      (hshape : (objects r).metadata.shape=[p.batches,p.heads,p.context,p.headDim])
      (active : z<p.batches ∧ h<p.heads ∧ n<p.context ∧ d<p.headDim) :
      ((z*p.heads+h)*p.context+n)*p.headDim+d < (tensors r).metadata.numel := by
    have hs : (tensors r).metadata.numel=p.batches*p.heads*p.context*p.headDim := by
      unfold Torch.TensorMetadata.numel
      rw [←view.shape r hr,hshape]
      simp [Nat.mul_assoc]
    rw [hs]
    exact flatten_lt _ _ _ _ _ _ _ _ active.1 active.2.1 active.2.2.1 active.2.2.2
  cases r
  · exact matrix .query (by decide) hinput.2.1 ha
  · exact matrix .key (by decide) (hinput.2.2.1.trans hinput.2.1) ha
  · exact matrix .value (by decide) (hinput.2.2.2.1.trans hinput.2.1) ha
  · exact False.elim (hr rfl)
  · change n<(tensors .queryScale).metadata.numel
    rw [←view.numel .queryScale (by decide)]
    exact lt_of_lt_of_le ha.2.2.2 hl.2.2.1
  · change n<(tensors .keyScale).metadata.numel
    rw [←view.numel .keyScale (by decide)]
    exact lt_of_lt_of_le ha.2.2.2 hl.2.2.2.1

def logicalRead (p : Config) (memory : ByteMemory) (tensors : Role→Torch.FP32.Tensor)
    (view : View p objects tensors) (r : Role) (hr : r≠Role.output) (z h n d : Nat) : BitVec 32 :=
  (Torch.FP32.logicalValue (fun address=>⟨readFP32 memory address⟩) (tensors r)
    (view.laws r hr) (logicalIndex p r z h n d)).bits

theorem descriptor_logical (p : Config) (objects : Role→TensorDescriptor)
    (tensors : Role→Torch.FP32.Tensor) (view : View p objects tensors) (memory : ByteMemory)
    (r : Role) (hr : r≠Role.output) (notNeg : (objects r).metadata.negative=Bool.false)
    (z h n d : Nat) (ha : roleActive p z h n d r)
    (hi : logicalIndex p r z h n d < (tensors r).metadata.numel) :
    descriptorRead objects memory r z h n d=logicalRead p memory tensors view r hr z h n d := by
  have hx := view.strided r hr z h n d ha
  have hn : (tensors r).metadata.isNeg=Bool.false := (view.negative r hr).symm.trans notNeg
  have hb := (view.laws r hr).inBounds _ hi
  rw [hx] at hb
  have hs := (tensors r).storage.spanFits
  have bound : (tensors r).storage.address (tensors r).storageOffset < 2^64 := by
    simp only [Torch.FP32.Storage.address]
    omega
  have hp := (view.laws r hr).pointer (by omega)
  have hpn := congrArg BitVec.toNat hp
  simp only [BitVec.toNat_ofNat,Nat.mod_eq_of_lt bound] at hpn
  unfold descriptorRead logicalRead Torch.FP32.logicalValue
  simp only [hn,Bool.false_eq_true,↓reduceIte]
  rw [view.pointer r hr,hpn,hx]
  simp only [Torch.FP32.Storage.address,Nat.mul_add,Nat.add_assoc]
  rfl

#axiomsClean valid_index
#axiomsClean descriptor_logical
end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.TensorBinding
