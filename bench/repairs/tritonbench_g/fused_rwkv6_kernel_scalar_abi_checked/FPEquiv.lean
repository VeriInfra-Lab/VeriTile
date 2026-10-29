import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.NativeScalar

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.FPEquiv
open VeriTile Triton FP.Structural FusedRwkv6Kernel
open scoped VeriTile.Spec
noncomputable section

/-- Issue the unchanged U load earlier in the prologue, before unrelated pointer
construction and the optional initial-state read. `raw` selects the complete
FP32/I32/I64/U64 or integer-1 source body in both schedules; casts and implicit
integer arithmetic conversion remain in those bodies. IO retains all input and
output channels, masks and the full grid from `Interface.programIO`.
No numeric operation changes, input restriction or experimental rule is added. -/
specification u_early_equiv (plan : Native.ReductionPlan) (binding : Native.ScalarBinding)
    (raw : PythonNumber) (c : Shape) (hc : Sizes c) :
    Runtime.original plan binding raw c ≡[[]]
      Runtime.earlyU plan binding raw c :=
  ProgramIO.Implementation.of_equivalent
    (Runtime.variant (Native.scalarAlgebra plan binding) c (Runtime.nativeScalar binding raw))
    (Runtime.equivalent (Native.scalarAlgebra plan binding) c hc (Native.plans c plan)
      (Native.scalar_rows plan binding c) (Runtime.nativeScalar binding raw))

#axiomsClean u_early_equiv
#print_fp_assumptions u_early_equiv
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.FPEquiv
