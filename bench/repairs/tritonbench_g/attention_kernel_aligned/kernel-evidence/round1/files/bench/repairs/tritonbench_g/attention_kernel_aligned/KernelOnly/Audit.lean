import bench.repairs.tritonbench_g.attention_kernel_aligned.Contract
import bench.repairs.tritonbench_g.attention_kernel_aligned.Correct
import bench.repairs.tritonbench_g.attention_kernel_aligned.FPEquiv
namespace VeriTile.Bench.Repairs.AttentionKernelAligned
open VeriTile Triton FP.Structural
#check KernelContract.accepted_iff
#check KernelContract.kernel_access
#check Correct.kernel_correct
#check BF16.Correct.kernel_correct
#check FPEquiv.kernel_native
#check BF16.FPEquiv.kernel_native
#check Correct.empty_kernel
#check BF16.Correct.empty_kernel
example (p : Full.Parameters) (x : Native.Expr) : (Native.algebraFor p).fp32Load x=x := rfl
example (p : Full.Parameters) (x : BF16.Native.Expr) : (BF16.Native.algebraFor p).fp32Load x=x := rfl
#axiomsClean KernelOnly.actual_address
#axiomsClean KernelOnly.bias_read
#axiomsClean KernelOnly.kernel_access
#axiomsClean KernelOnly.complete
#axiomsClean BF16.KernelOnly.complete
#axiomsClean KernelOnly.backing_exterior
#axiomsClean BF16.KernelOnly.backing_exterior
#axiomsClean KernelOnly.source_algorithm
#axiomsClean BF16.KernelOnly.source_algorithm
#axiomsClean Correct.kernel_correct
#axiomsClean BF16.Correct.kernel_correct
#axiomsClean FPEquiv.kernel_native
#axiomsClean BF16.FPEquiv.kernel_native
#axiomsClean Correct.empty_kernel
#axiomsClean BF16.Correct.empty_kernel
#specNonCircular KernelOnly.mathematicalTarget avoiding [FP.Structural.exec,Whole.runPrograms,Whole.answer,Source.rounds,Source.advance]
#specNonCircular KernelOnly.nativeReference avoiding [FP.Structural.exec,Whole.runPrograms,Whole.answer,Source.rounds,Source.advance]
#specNonCircular BF16.KernelOnly.mathematicalTarget avoiding [FP.Structural.exec,BF16.Whole.runPrograms,BF16.Whole.answer,BF16.Source.rounds,BF16.Source.advance]
#specNonCircular BF16.KernelOnly.nativeReference avoiding [FP.Structural.exec,BF16.Whole.runPrograms,BF16.Whole.answer,BF16.Source.rounds,BF16.Source.advance]
#specNonCircular KernelOnly.InputContract avoiding [FP.Structural.exec,Whole.runPrograms,BF16.Whole.runPrograms]
end VeriTile.Bench.Repairs.AttentionKernelAligned
