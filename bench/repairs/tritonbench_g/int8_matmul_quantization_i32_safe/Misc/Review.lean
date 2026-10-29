import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Correct
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Misc.NativeCorrect
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.FPEquiv
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Misc.TensorBridge
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.CompositionIO
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.IntegerDotModeChecks

/-! Reader/audit aggregator. Supporting proof modules do not import this file
or the public correctness/equivalence readers. -/
namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe

#axiomsClean KernelOnly.Correct.producer_correct
#axiomsClean KernelOnly.Correct.gemm_correct
#axiomsClean KernelOnly.FPEquiv.producer_native
#axiomsClean KernelOnly.FPEquiv.gemm_native
#axiomsClean PublicIO.Composition.composed_real
#axiomsClean PublicIO.Composition.composed_native
#axiomsClean PublicIO.Gemm.source_at_arbitrary_prefix

end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe
