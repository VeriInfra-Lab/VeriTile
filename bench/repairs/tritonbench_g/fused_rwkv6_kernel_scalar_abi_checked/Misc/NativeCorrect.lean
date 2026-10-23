import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Misc.NativeIOResults
import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Misc.RawNative

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.FPEquiv
open VeriTile Triton FP.Structural FusedRwkv6Kernel
noncomputable section

def kernelIO (p : Native.ReductionPlan) (binding : Native.ScalarBinding)
    (raw : PythonNumber) (c : Shape) : ProgramIO Native.Term Native.Term :=
  Interface.programIO (Native.algebra p) c (Native.scalar binding raw)
abbrev formula := Reference.native

/-- Source-to-native observation of the full register loop and original scalar
ABI. Primitive, contraction and shape-only reduction choices remain explicit;
this theorem introduces no floating transformation or numerical experiment law. -/
theorem rwkv_native (a : Native.Instructions) (compiler : Native.Compiler)
    (p : Native.ReductionPlan) (binding : Native.ScalarBinding) (raw : PythonNumber)
    (c : Shape) (hc : Sizes c) :
    (kernelIO p binding raw c).Observes (Reference.nativeObservation a compiler raw c)
      (formula a compiler p binding raw c) :=
  Interface.Result.native a compiler p binding raw c hc

#stmtSurfaceSubset rwkv_native ⊆
  [Native.Instructions,Native.Compiler,Native.ReductionPlan,Native.ScalarBinding,
   PythonNumber,Shape,Sizes,Native.Term,Native.instInhabitedTerm,
   ProgramIO.Observes,kernelIO,Reference.nativeObservation,formula]
#specNonCircular formula avoiding [kernel,Source.launch,FP.Structural.exec,FusedRwkv6Kernel.runProgram]
#axiomsClean rwkv_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.FPEquiv
