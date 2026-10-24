import bench.repairs.tritonbench_g.softmax_triton1.Proofs.Contract
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.Native
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.SoftmaxTriton1.Contiguous
open VeriTile.Triton
noncomputable section

/-- Actual Tensor object and its ordinary FP32 storage binding. The region
names its backing allocation, not a copied contiguous snapshot. -/
structure Object where
  tensor : VeriTile.Torch.FP32.Tensor
  laws : VeriTile.Torch.FP32.TensorLaws tensor

/-- Backing-region identity is derived from the actual storage object. Aliased
Tensors cannot be assigned unrelated source memories. -/
def Object.region (x : Object) : RegionName := ⟨"torch-storage:" ++ toString x.tensor.storage.object⟩

theorem Object.same_storage_region (x y : Object) (same : x.tensor.storage=y.tensor.storage) :
    x.region=y.region := by simp [Object.region,same]

abbrev Object.metadata (x : Object) := x.tensor.metadata

def descriptor (x : Object) : Tensor where
  objectId := x.tensor.object
  region := x.region
  rows := rows x.metadata
  cols := cols x.metadata
  offset := x.tensor.storageOffset
  rowStride := cols x.metadata
  colStride := 1
  device := x.metadata.device.index
  negative := x.metadata.isNeg

def objectValue {S C V} (a : Access S C V) (neg : V → V)
    (s : S) (x : Object) (r c : Nat) : V :=
  let v := a.read s x.region (x.tensor.physicalIndex (r*cols x.metadata+c))
  if x.metadata.isNeg then neg v else v

theorem descriptor_injective (x : Object) : (descriptor x).Injective := by
  intro r c r' c' _ hc _ hc' eq
  change c<cols x.metadata at hc
  change c'<cols x.metadata at hc'
  simp only [Tensor.index,descriptor,Nat.mul_one] at eq
  have rows : r=r' := VeriTile.Bench.Repairs.SquareMatrix.row_unique _ r' c' r hc'
    (by omega) (by omega)
  exact ⟨rows,by subst r';omega⟩


/-- The source region is the actual Tensor backing storage. Every loaded cell
is interpreted by one physical-address reader, not an unrelated input array. -/
def InputMemory {S C V} (a : Access S C V) (read : Nat → V)
    (x : Object) (s : S) : Prop :=
  ∀ i, i < x.tensor.storage.elements →
    a.read s x.region i = read (x.tensor.storage.address i)

/-- Independent logical input values at original Tensor coordinates. -/
def logicalValue {V} (neg : V → V) (read : Nat → V) (x : Object) (r c : Nat) : V :=
  let value := read (x.tensor.storage.address
    (x.tensor.physicalIndex (r * cols x.metadata + c)))
  if x.metadata.isNeg then neg value else value

theorem input_read {S C V} (a : Access S C V) (neg : V → V) (read : Nat → V)
    (x : Object) (h : InputContract x.metadata) (s : S) (binding : InputMemory a read x s)
    (r c : Nat) (hr : r < rows x.metadata) (hc : c < cols x.metadata) :
    a.read s x.region (x.tensor.storageOffset + r * cols x.metadata + c) =
      logicalValue neg read x r c := by
  have hi : r * cols x.metadata + c < x.metadata.numel := by
    rw [numel_eq _ h]
    exact VeriTile.Bench.Repairs.SquareMatrix.flat_lt _ _ _ _ hr hc
  have physical := x.laws.contiguous h.contiguous _ hi
  have bound := x.laws.inBounds _ hi
  rw [physical] at bound
  simpa [logicalValue,h.noNeg,physical,Nat.add_assoc] using binding _ bound

#axiomsClean input_read

end
end VeriTile.Bench.Repairs.SoftmaxTriton1.Contiguous
