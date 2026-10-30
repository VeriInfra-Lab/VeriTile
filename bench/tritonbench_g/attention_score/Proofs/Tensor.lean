import VeriTile.Torch.FP32
import bench.tritonbench_g.attention_score.Proofs.Reused.Contract

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked
open VeriTile Triton Reused Geometry Physical
noncomputable section

def inputRoles : List RegionName := ["q", "k", "m"]
def tensorShape (c : Shape) (r : RegionName) : List Nat :=
  if r="q" then [c.Z,c.H,c.N,c.D] else if r="k" then [c.Z,c.HK,c.NK,c.D]
  else if r="m" then [c.Z,c.H,c.R] else [c.Z,c.H,c.NK]

/-- Observed objects and ordinary storage/API laws. Shape matching is the
binding of the checked dimensions to those objects, not an output law. -/
structure Tensors (c : Shape) where
  get : RegionName → Torch.FP32.Tensor
  ordinary : RegionName → Bool
  conjugate : RegionName → Bool
  api : ∀ r, tracked r → Torch.FP32.TensorLaws (get r)
  shape : ∀ r, tracked r → (get r).metadata.shape = tensorShape c r
  outputContiguous : (get "out").metadata.isContiguous = Bool.true
  outputNotNeg : (get "out").metadata.isNeg = Bool.false

/-- Exactly the input metadata queried by the selected Python wrapper. -/
def observations (t : Tensors c) : Reused.Contract.Observations where
  ordinaryTensors := inputRoles.all (fun r => t.ordinary r)
  cudaFP32 := inputRoles.all (fun r => (t.get r).metadata.isCuda && (t.get r).metadata.isFP32)
  oneDevice := inputRoles.all (fun r => decide ((t.get r).metadata.device = (t.get "q").metadata.device))
  contiguous := inputRoles.all (fun r => (t.get r).metadata.isContiguous)
  strided := inputRoles.all (fun r => (t.get r).metadata.isStrided)
  resolvedViews := inputRoles.all (fun r => !(t.get r).metadata.isNeg && !(t.conjugate r))
  pointer := fun r => (t.get r).dataPtr.toNat

theorem Tensors.count (t : Tensors c) (r : RegionName) (hr : tracked r) :
    (t.get r).metadata.numel = Physical.size c r := by
  rw [Torch.TensorMetadata.numel,t.shape r hr]
  rcases hr with rfl|rfl|rfl|rfl <;>
    simp [tensorShape,Physical.size,Shape.querySize,Shape.keySize,Shape.normalizerSize,
      Shape.outputSize,Nat.mul_assoc]

theorem Tensors.contiguous (t : Tensors c) (h : Reused.Contract.Metadata (observations t))
    (r : RegionName) (hr : tracked r) : (t.get r).metadata.isContiguous = Bool.true := by
  rcases hr with rfl|rfl|rfl|rfl
  all_goals try exact t.outputContiguous
  all_goals have hc := h.2.2.2.1
  all_goals simp only [observations,inputRoles,List.all_cons,List.all_nil,Bool.and_true,Bool.and_eq_true] at hc
  · exact hc.1
  · exact hc.2.1
  · exact hc.2.2

theorem Tensors.notNeg (t : Tensors c) (h : Reused.Contract.Metadata (observations t))
    (r : RegionName) (hr : tracked r) : (t.get r).metadata.isNeg = Bool.false := by
  rcases hr with rfl|rfl|rfl|rfl
  all_goals try exact t.outputNotNeg
  all_goals have hs := h.2.2.2.2.2
  all_goals simp only [observations,inputRoles,List.all_cons,List.all_nil,Bool.and_true,
    Bool.and_eq_true] at hs
  · simpa using congrArg Bool.not hs.1.1
  · simpa using congrArg Bool.not hs.2.1.1
  · simpa using congrArg Bool.not hs.2.2.1

/-- Logical address derives from core Tensor capacity and contiguous API
semantics; nonzero storage offsets and overlapping read tensors are allowed. -/
theorem Tensors.address (t : Tensors c) (h : Reused.Contract.Metadata (observations t))
    (r : RegionName) (hr : tracked r) (i : Nat) (hi : i < Physical.size c r) :
    (t.get r).storage.address ((t.get r).physicalIndex i) = (observations t).pointer r+4*i := by
  let x := t.get r
  have laws := t.api r hr
  have bound : i < x.metadata.numel := by simpa [x,t.count r hr] using hi
  have cont := t.contiguous h r hr
  have cap := Torch.FP32.contiguous_capacity x laws cont (by omega)
  have span := x.storage.spanFits
  have baseBound : x.storage.address x.storageOffset < 2^64 := by
    unfold Torch.FP32.Storage.address
    omega
  rw [laws.contiguous cont i bound]
  have ptr := laws.pointer (Nat.zero_lt_of_lt bound)
  have ptrNat := congrArg BitVec.toNat ptr
  rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt baseBound] at ptrNat
  change x.dataPtr.toNat = x.storage.address x.storageOffset at ptrNat
  change x.storage.address (x.storageOffset+i) = x.dataPtr.toNat+4*i
  rw [ptrNat]
  simp [Torch.FP32.Storage.address,Nat.mul_add,Nat.add_assoc]

def logical {α : Type} (read : Nat → α) (t : Tensors c) (r : RegionName) (i : Nat) : α :=
  read ((t.get r).storage.address ((t.get r).physicalIndex i))

theorem logical_direct {α : Type} (read : Nat → α) (t : Tensors c)
    (h : Reused.Contract.Metadata (observations t)) (r : RegionName) (hr : tracked r)
    (i : Nat) (hi : i < Physical.size c r) :
    logical read t r i = read ((observations t).pointer r+4*i) := by
  rw [logical,t.address h r hr i hi]

theorem logical_fp32 (read : Nat → Torch.FP32.Word) (t : Tensors c)
    (h : Reused.Contract.Metadata (observations t)) (r : RegionName) (hr : tracked r)
    (i : Nat) (hi : i < Physical.size c r) :
    Torch.FP32.logicalValue read (t.get r) (t.api r hr) i = read ((observations t).pointer r+4*i) := by
  simp only [Torch.FP32.logicalValue,t.notNeg h r hr,Bool.false_eq_true,↓reduceIte]
  exact logical_direct read t h r hr i hi

#axiomsClean Tensors.address
#axiomsClean logical_fp32
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked
