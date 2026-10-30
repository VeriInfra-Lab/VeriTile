import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Auxiliary
/-- Exact previous operational endpoints; public IO results live in Correct/FPEquiv. -/
alias raw_kernel_correctness := RawKernel.real
alias empty_kernel := RawKernel.empty_kernel
alias native_kernel_observation := RawKernel.native
noncomputable abbrev kernel_reference := RawKernel.nativeReference
#axiomsClean raw_kernel_correctness
#axiomsClean empty_kernel
#axiomsClean native_kernel_observation
end VeriTile.Bench.Repairs.AttentionKernelAligned.Auxiliary

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Auxiliary
/-- Exact previous operational endpoints; public IO results live in Correct/FPEquiv. -/
alias raw_kernel_correctness := RawKernel.real
alias empty_kernel := RawKernel.empty_kernel
alias native_kernel_observation := RawKernel.native
noncomputable abbrev kernel_reference := RawKernel.nativeReference
#axiomsClean raw_kernel_correctness
#axiomsClean empty_kernel
#axiomsClean native_kernel_observation
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Auxiliary

