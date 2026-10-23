import bench.repairs.tritonbench_g.attention_kernel.Contract
import bench.repairs.tritonbench_g.attention_kernel.Correct
import bench.repairs.tritonbench_g.attention_kernel.Misc.NativeCorrect
import bench.repairs.tritonbench_g.attention_kernel.Misc.TensorBridge

namespace VeriTile.Bench.Repairs.AttentionKernel.KernelOnly
open VeriTile Triton FP.Structural Whole001 Source Whole Whole002
#check KernelContract.accepted_iff
#check Correct.kernel_correct
#check FPEquiv.kernel_native
#check KernelResults.Real.empty_kernel
example {α : Type} (M : Algebra α) (D : Dot α) : (withDot M D).cast=M.cast := rfl
example (layout : FamilyNative.Layout) (x : FamilyNative.Expr) :
    (FamilyNative.algebra layout).fp32Load x=x := rfl
#axiomsClean check_iff
#axiomsClean actual_address
#axiomsClean actual_read
#axiomsClean bias_read
#axiomsClean complete
#axiomsClean source_algorithm
#axiomsClean Correct.kernel_correct
#axiomsClean FPEquiv.kernel_native
#axiomsClean KernelResults.Real.empty_kernel
#specNonCircular mathematicalTarget avoiding [FP.Structural.exec,Whole.runPrograms,Whole.answer,Source.accumulated,Source.update]
#specNonCircular nativeReference avoiding [FP.Structural.exec,Whole.runPrograms,Whole.answer,Source.accumulated,Source.update]
#specNonCircular InputContract avoiding [FP.Structural.exec,Whole.runPrograms,Whole.answer]
end VeriTile.Bench.Repairs.AttentionKernel.KernelOnly
