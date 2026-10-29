import bench.tritonbench_g.relu_strided_buffer.Kernels
import bench.tritonbench_g.relu_strided_buffer.Proofs.SignedByteAddress

namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.Whole
open VeriTile.Triton HostPlan SignedView

/-- Both source constructor offset branches, both optional override branches
and every ordinary metadata method. tensor is passed directly to runHost;
there is no second view or remapping of logical tensor coordinates. -/
theorem wrapped_metadata (base : Buffer) (shape : Option (List Nat))
    (strides : Option (List Int)) (offset : Int) (allocationBase : Nat) :
    let x := wrapBuffer base shape strides offset
    x.tensor = reinterpret base shape strides offset ∧
    x.tensor.region = base.region ∧ x.tensor.deviceIndex = base.deviceIndex ∧
    x.dtype = .fp32 ∧ x.stride = strides.getD base.strides ∧
    x.size = shape.getD base.shape ∧ x.elementSize = 4 ∧
    x.numel = (shape.getD base.shape).prod ∧
    x.dim = (shape.getD base.shape).length ∧ x.unwrap = base ∧
    x.dataPtr allocationBase = ((allocationBase : Int) + 4 * base.base) + 4 * offset := by
  simp [wrapBuffer, reinterpret, WrappedBuffer.dtype, WrappedBuffer.stride,
    WrappedBuffer.size, WrappedBuffer.elementSize, WrappedBuffer.numel,
    WrappedBuffer.dim, WrappedBuffer.unwrap, WrappedBuffer.dataPtr, Int.mul_add, Int.add_assoc]

theorem zero_offset_pointer (base : Buffer) (shape : Option (List Nat))
    (strides : Option (List Int)) (allocationBase : Nat) :
    (wrapBuffer base shape strides 0).dataPtr allocationBase =
      (allocationBase : Int) + 4 * base.base := by
  simp [wrapBuffer, reinterpret, WrappedBuffer.dataPtr]

/-- The host constructor's byte-valued data_ptr and the finite argument used
by the same source kernel coincide for every nonempty valid view. -/
theorem wrapped_pointer_exact (base : Buffer) (shape : Option (List Nat))
    (strides : Option (List Int)) (offset stride : Int) (allocationBase : Nat)
    {n capacity : Nat} (positive : 0 < n)
    (valid : ((wrapBuffer base shape strides offset).tensor.view stride).Valid n capacity)
    (allocation : SignedByteAddress.AllocationDomain allocationBase capacity) :
    (SignedByteAddress.dataPtr64 allocationBase
      ((wrapBuffer base shape strides offset).tensor.view stride)).toNat =
      ((wrapBuffer base shape strides offset).dataPtr allocationBase).toNat := by
  rw [SignedByteAddress.data_ptr_exact allocationBase _ valid allocation positive]
  have nonneg := (valid 0 positive).1
  simp only [Buffer.view, View.signedAddress, Nat.cast_zero, zero_mul, add_zero] at nonneg
  simp [WrappedBuffer.dataPtr, Buffer.view, Int.toNat_add, Int.toNat_mul, nonneg]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.Whole
