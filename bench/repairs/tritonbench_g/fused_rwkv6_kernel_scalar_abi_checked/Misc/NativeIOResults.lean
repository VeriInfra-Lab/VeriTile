import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.IOResults
import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Misc.NativeReference

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Interface.Result
open VeriTile Triton FP.Structural FusedRwkv6Kernel
noncomputable section

theorem native (a : Native.Instructions) (compiler : Native.Compiler) (p : Native.ReductionPlan)
    (binding : Native.ScalarBinding) (raw : PythonNumber) (c : Shape) (hc : Sizes c) :
    (programIO (Native.algebra p) c (Native.scalar binding raw)).Observes
      (Reference.nativeObservation a compiler raw c) (Reference.native a compiler p binding raw c) := by
  have h := ProgramIO.Observes.of_implements
    (Proof.implements (Native.algebra p) c hc (Native.plans c p) (Source.native_rows p c)
      (Native.scalar binding raw)) (Reference.nativeObservation a compiler raw c)
  simpa only [Reference.native,Native.source_ops,ProgramIO.mapOutputs,Reference.nativeObservation] using h

#axiomsClean native
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Interface.Result
