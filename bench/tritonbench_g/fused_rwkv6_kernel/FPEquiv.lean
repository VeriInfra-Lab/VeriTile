import bench.tritonbench_g.fused_rwkv6_kernel.Proofs.Pair

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.FPEquiv
open VeriTile Triton FP.Structural FusedRwkv6Kernel
open scoped VeriTile.Spec
noncomputable section

/-- Issue the unchanged U load earlier in the prologue, before unrelated pointer
construction and the optional initial-state read. No numeric operation changes. -/
specification u_early_equiv (plan : Native.ReductionPlan) (binding : Native.ScalarBinding)
    (raw : PythonNumber) (c : Shape) (hc : Sizes c) :
    Pair.original plan c (Native.scalar binding raw) ≡[[]]
      Pair.earlyU plan c (Native.scalar binding raw) :=
  ProgramIO.Implementation.of_equivalent
    (Pair.variant (Native.algebra plan) c (Native.scalar binding raw))
    (Pair.equivalent (Native.algebra plan) c hc (Native.plans c plan)
      (Source.native_rows plan c) (Native.scalar binding raw))

#axiomsClean u_early_equiv
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.FPEquiv
