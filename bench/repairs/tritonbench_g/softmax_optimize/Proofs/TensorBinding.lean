import bench.repairs.tritonbench_g.softmax_optimize.Proofs.Pointers
import bench.repairs.tritonbench_g.softmax_optimize.Kernels
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.Binding
open VeriTile Triton

structure Object where
  tensor : VeriTile.Torch.FP32.Tensor
  laws : VeriTile.Torch.FP32.TensorLaws tensor

abbrev Object.metadata (x : Object) := x.tensor.metadata
def Object.region (x : Object) : RegionName := ⟨"torch-storage:" ++ toString x.tensor.storage.object⟩

def logicalValue {V} (neg : V → V) (read : Nat → V) (x : Object)
    (row : Fin (rows x.metadata)) (column : Fin (columns x.metadata)) : V :=
  let v := read (x.tensor.storage.address (x.tensor.physicalIndex
    (row.val*columns x.metadata+column.val)))
  if x.metadata.isNeg then neg v else v

def MemoryBinding {S V} (value : S → RegionName → Nat → V) (read : Nat → V)
    (x : Object) (s : S) : Prop :=
  ∀ i,i<x.tensor.storage.elements → value s x.region i=read (x.tensor.storage.address i)

theorem flat_lt (x : Object) (h : InputContract x.metadata)
    (row : Fin (rows x.metadata)) (column : Fin (columns x.metadata)) :
    row.val*columns x.metadata+column.val<x.metadata.numel := by
  rw [count_two x.metadata h.2.2.2.1]
  exact Native.active_element_bound row column

theorem span (x : Object) (h : InputContract x.metadata) (live : 0<x.metadata.numel) :
    x.metadata.dataPtr.toNat+4*x.metadata.numel≤2^64 := by
  have cap := VeriTile.Torch.FP32.contiguous_capacity x.tensor x.laws h.2.2.2.2.1 live
  change x.tensor.storageOffset+x.metadata.numel≤x.tensor.storage.elements at cap
  have total := x.tensor.storage.spanFits
  have pointer := x.laws.pointer live
  have first : x.tensor.storage.address x.tensor.storageOffset<2^64 := by
    unfold VeriTile.Torch.FP32.Storage.address
    omega
  change x.metadata.dataPtr=_ at pointer
  rw [pointer,BitVec.toNat_ofNat,Nat.mod_eq_of_lt first]
  unfold VeriTile.Torch.FP32.Storage.address
  omega

theorem address (x : Object) (h : InputContract x.metadata)
    (row : Fin (rows x.metadata)) (column : Fin (columns x.metadata)) :
    x.metadata.dataPtr.toNat+4*(row.val*columns x.metadata+column.val)=
      x.tensor.storage.address (x.tensor.physicalIndex (row.val*columns x.metadata+column.val)) := by
  have hi := flat_lt x h row column
  have live : 0<x.metadata.numel := by omega
  have physical := x.laws.contiguous h.2.2.2.2.1 _ hi
  have cap := VeriTile.Torch.FP32.contiguous_capacity x.tensor x.laws h.2.2.2.2.1 live
  change x.tensor.storageOffset+x.metadata.numel≤x.tensor.storage.elements at cap
  have total := x.tensor.storage.spanFits
  have first : x.tensor.storage.address x.tensor.storageOffset<2^64 := by
    unfold VeriTile.Torch.FP32.Storage.address
    omega
  have pointer := x.laws.pointer live
  change x.metadata.dataPtr=_ at pointer
  rw [pointer,BitVec.toNat_ofNat,Nat.mod_eq_of_lt first,physical]
  simp [VeriTile.Torch.FP32.Storage.address,Nat.mul_add,Nat.add_assoc]

/-- Actual native pointer, physical storage element and original logical
Tensor coordinate are connected before values enter either kernel proof. -/
theorem source_load {S V} (value : S → RegionName → Nat → V) (neg : V → V)
    (read : Nat → V) (x : Object) (h : InputContract x.metadata) (s : S)
    (binding : MemoryBinding value read x s)
    (row : Fin (rows x.metadata)) (column : Fin (columns x.metadata)) :
    value s x.region (x.tensor.storageOffset+row.val*columns x.metadata+column.val)=
      read (Native.nativeByte x.metadata.dataPtr.toNat (columns x.metadata) row column 0) ∧
    read (Native.nativeByte x.metadata.dataPtr.toNat (columns x.metadata) row column 0)=
      logicalValue neg read x row column := by
  have hi := flat_lt x h row column
  have physical := x.laws.contiguous h.2.2.2.2.1 _ hi
  have bound := x.laws.inBounds _ hi
  have ptr := (Pointers.input_bytes x.metadata h (span x h (by omega)) row column 0).1
  simp only [Fin.val_zero,Nat.add_zero] at ptr
  rw [ptr,address x h row column]
  constructor
  · simpa [physical,Nat.add_assoc] using binding _ bound
  · simp [logicalValue,h.2.2.2.2.2.1]

#axiomsClean span
#axiomsClean address
#axiomsClean source_load
end VeriTile.Bench.Repairs.SoftmaxOptimize.Binding
