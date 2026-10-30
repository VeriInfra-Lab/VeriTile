import bench.repairs.tritonbench_g.softmax_triton2.Contract
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.Addresses

namespace VeriTile.Bench.Repairs.SoftmaxTriton2.Binding
open VeriTile.Triton
open VeriTile.Bench.Repairs.SoftmaxTriton1 Contiguous
noncomputable section

/-- Proof reuse is tied to literal equality of the two typed source ASTs. -/
theorem source_identity (out inp : RegionName) (ib ob ir ic orow oc n B : Nat) :
    SoftmaxTriton2.softmax_kernel out inp ib ob ir ic orow oc n B =
      SoftmaxTriton1.softmax_kernel out inp ib ob ir ic orow oc n B := rfl

/-- A source load addresses the same original Tensor cell as its logical
coordinate. Both the instruction's pointer chain and the reader are explicit;
there is no copied input array. Only active lanes need a storage read. -/
theorem active_load {S C V} (a : Access S C V) (neg : V → V)
    (read : Nat → V) (x : Object) (h : InputContract x.metadata)
    (s : S) (binding : InputMemory a read x s)
    (r c : Nat) (hr : r < rows x.metadata) (hc : c < cols x.metadata) :
    a.read s x.region ((descriptor x).index r c) =
        read (Native.sequentialSourcePointer (buffer x) (descriptor x) r c).toNat ∧
      a.read s x.region ((descriptor x).index r c) = logicalValue neg read x r c ∧
      (Native.sequentialSourcePointer (buffer x) (descriptor x) r c).toNat =
        x.tensor.storage.address (x.tensor.physicalIndex (r * cols x.metadata + c)) ∧
      (Native.sequentialSourcePointer (buffer x) (descriptor x) r c).toNat + 4 ≤
        x.tensor.storage.base + 4 * x.tensor.storage.elements ∧
      (Native.sequentialSourcePointer (buffer x) (descriptor x) r c).toNat % 4 = 0 := by
  have value : a.read s x.region ((descriptor x).index r c) = logicalValue neg read x r c := by
    simpa [descriptor,Tensor.index] using input_read a neg read x h s binding r c hr hc
  have pointer := pointer_bounds x h r c hr hc
  refine ⟨?_,value,pointer⟩
  rw [value,pointer.1]
  simp [logicalValue,h.noNeg]

/-- Actual Tensor contiguity connects the output descriptor to physical
storage as well, including nonzero offsets and singleton metadata strides. -/
theorem physical_index (x : Object) (h : InputContract x.metadata)
    (r c : Nat) (hr : r < rows x.metadata) (hc : c < cols x.metadata) :
    (descriptor x).index r c = x.tensor.physicalIndex (r * cols x.metadata + c) := by
  have hi : r * cols x.metadata + c < x.metadata.numel := by
    rw [numel_eq _ h]
    exact VeriTile.Bench.Repairs.SquareMatrix.flat_lt _ _ _ _ hr hc
  simp [descriptor,Tensor.index,x.laws.contiguous h.contiguous _ hi,Nat.add_assoc]

#axiomsClean source_identity
#axiomsClean active_load
#axiomsClean physical_index
end
end VeriTile.Bench.Repairs.SoftmaxTriton2.Binding
