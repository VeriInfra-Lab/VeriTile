import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.RawAPI
import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.KernelTensor

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4
open VeriTile.Triton

#specNonCircular Checked.InputContract avoiding [KernelSource.run,Native.grid,Physical.source]
#specNonCircular Real.mathematical avoiding [KernelSource.run,RealHost.grid,Real.value]
#specNonCircular Logical.output avoiding [Native.exec,Native.grid,Grid.grid,Physical.source]
#specNonCircular KernelTensor.RealPost avoiding [KernelSource.run,KernelSource.one,RealHost.grid,Real.value]
#specNonCircular KernelTensor.NativePost avoiding [Native.exec,Native.grid,Grid.grid,Physical.source]
#axiomsClean Checked.checkInputs_iff
#axiomsClean Checked.checkArguments_iff
#axiomsClean KernelTensor.machine_safe
#axiomsClean KernelIndex.program_id
#axiomsClean KernelIndex.formed_addresses
#axiomsClean Physical.commit_write
#axiomsClean TensorBinding.store_address
#axiomsClean Correct.kernel_correct_raw
#axiomsClean FPEquiv.kernel_equiv_raw
#axiomsClean Correct.quantize_raw
#axiomsClean FPEquiv.quantize_raw
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.MatmulDequantizeInt4
