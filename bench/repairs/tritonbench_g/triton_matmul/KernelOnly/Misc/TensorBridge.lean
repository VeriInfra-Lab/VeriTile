import bench.repairs.tritonbench_g.triton_matmul.KernelOnly.Misc.NativeCorrect
import bench.repairs.tritonbench_g.triton_matmul.Misc.KernelWhole

/-! Optional logical Tensor application and physical input-frame binding exports.
The primary mathematical/native kernel facade does not import this module. -/
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.TensorBridge
alias real_tensor := VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelTensor.real_tensor
alias input_bytes := VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelWhole.input_bytes
alias kernel_real := VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelCorrect.kernel_real
alias kernel_native_if_local := VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelFPEquiv.kernel_native_if_local
alias empty_programs := VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelSource.empty_programs
alias safety := VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelSource.safety
#axiomsClean kernel_real
#axiomsClean kernel_native_if_local
#axiomsClean empty_programs
#axiomsClean safety
#axiomsClean real_tensor
#axiomsClean input_bytes
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.TensorBridge
