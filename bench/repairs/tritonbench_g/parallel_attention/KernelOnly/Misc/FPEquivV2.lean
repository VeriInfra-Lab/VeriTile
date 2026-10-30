import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.FPEquiv
import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.Misc.NativeCorrect

/-! Compatibility reader for the complete forward/backward native IO
observations. The public IO record and specifications live in FPEquiv;
actual Tensor/common-backing applications remain in TensorBridge. -/
