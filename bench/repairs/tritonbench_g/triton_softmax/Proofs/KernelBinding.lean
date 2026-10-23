import bench.repairs.tritonbench_g.triton_softmax.Contract
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.Addresses

namespace VeriTile.Bench.Repairs.TritonSoftmax.KernelBinding
open VeriTile.Triton
open VeriTile.Bench.Repairs.SoftmaxTriton1 Contiguous
noncomputable section

/-- Compose actual backing storage, the widened source byte pointer, and the
original logical Tensor coordinate before either execution proof uses a value. -/
theorem source_load {S C V} (a : Access S C V) (neg : V → V) (read : Nat → V)
    (x : Object) (h : InputContract x.metadata) (s : S) (binding : InputMemory a read x s)
    (r c : Nat) (hr : r < rows x.metadata) (hc : c < cols x.metadata) :
    a.read s x.region ((descriptor x).index r c) =
      read (Native.sequentialSourcePointer (buffer x) (descriptor x) r c).toNat ∧
    read (Native.sequentialSourcePointer (buffer x) (descriptor x) r c).toNat =
      logicalValue neg read x r c := by
  have logical := input_read a neg read x h s binding r c hr hc
  have physical := (pointer_bounds x h r c hr hc).1
  have pointer : read (Native.sequentialSourcePointer (buffer x) (descriptor x) r c).toNat =
      logicalValue neg read x r c := by
    simp only [physical,logicalValue,h.noNeg,Bool.false_eq_true,if_false]
  exact ⟨(by simpa [descriptor,Tensor.index] using logical.trans pointer.symm),pointer⟩

theorem physical_index (x : Object) (h : InputContract x.metadata)
    (r c : Nat) (hr : r < rows x.metadata) (hc : c < cols x.metadata) :
    x.tensor.physicalIndex (r*cols x.metadata+c) = (descriptor x).index r c := by
  have hi : r*cols x.metadata+c < x.metadata.numel := by
    rw [numel_eq _ h]
    exact VeriTile.Bench.Repairs.SquareMatrix.flat_lt _ _ _ _ hr hc
  rw [x.laws.contiguous h.contiguous _ hi]
  simp [descriptor,Tensor.index,Nat.add_assoc]

#axiomsClean source_load
#axiomsClean physical_index
end
end VeriTile.Bench.Repairs.TritonSoftmax.KernelBinding
