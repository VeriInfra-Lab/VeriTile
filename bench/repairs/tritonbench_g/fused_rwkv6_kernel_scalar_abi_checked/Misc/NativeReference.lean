import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.IO

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Reference
open VeriTile Triton FP.Structural FusedRwkv6Kernel
noncomputable section

def nativeObservation (a : Native.Instructions) (compiler : Native.Compiler)
    (raw : PythonNumber) (c : Shape) (_o : Fin (Interface.outputPorts c).length)
    (_j : TileIndex (Interface.outputPorts c)[_o].shape) : Native.Term → BitVec 32 :=
  Native.observeBound a compiler raw

def native (a : Native.Instructions) (compiler : Native.Compiler) (p : Native.ReductionPlan)
    (binding : Native.ScalarBinding) (raw : PythonNumber) (c : Shape)
    (xs : Interface.Inputs Native.Term c) : ChannelValues (BitVec 32) (Interface.outputPorts c) := fun o j =>
  mapValue (Native.observeBound a compiler raw) (Interface.outputPorts c)[o].dtype
    (Interface.formula Native.referenceOps c (Native.plans c p) (Native.scalar binding raw) xs o j)

#specNonCircular native avoiding [kernel,Source.launch,FP.Structural.exec,FusedRwkv6Kernel.runProgram]
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Reference
