import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Misc.NativeCorrect

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.FPEquiv
noncomputable section

/-- Compatibility names for the retained independent native formula and its
observed pair result, with their original domains and native observers. -/
abbrev nativeIO := NativeCorrect.nativeIO
alias rmsnorm_kernel_native := NativeCorrect.rmsnorm_kernel_native

end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.FPEquiv
