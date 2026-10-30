import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.RuntimeScalar
import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.LogicalCoordinates

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Runtime
open VeriTile Triton FP.Structural FusedRwkv6Kernel
noncomputable section

/-- All scalar source profiles satisfy the same independent Real formula;
the full logical input/output bijections and complete IO judgment are retained. -/
theorem logical_real (c : Shape) (hc : Sizes c) (scalar : Scalar ℝ) :
    (logicalIO c scalar).Implements (fun xs => {
      partials := fun ik z t f => Real.partialFormula c (Interface.tensorData c xs)
        (scalar.value Real.algebra) z.val ik.val (visit c t.val) f.val
      finalState := fun z e f => realState c (Interface.tensorData c xs) z.val e.val f.val c.T }) :=
  (implements_iff Real.algebra c scalar _).mpr
    (Interface.Result.logical_real c hc (scalar.value Real.algebra))

#axiomsClean logical_real
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Runtime
