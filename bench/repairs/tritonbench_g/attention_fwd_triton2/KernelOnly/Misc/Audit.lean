import bench.repairs.tritonbench_g.attention_fwd_triton2.Misc.NativeCorrect
import bench.repairs.tritonbench_g.attention_fwd_triton2.Contract
import bench.repairs.tritonbench_g.attention_fwd_triton2.Correct
import bench.repairs.tritonbench_g.attention_fwd_triton2.FPEquiv
import bench.repairs.tritonbench_g.attention_fwd_triton2.Misc.TensorBridge

namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.KernelOnly
open VeriTile Triton FP.Structural Whole001 Source Whole
#check KernelContract.accepted_iff
#check Correct.kernel_raw_correct
#check FPEquiv.kernel_raw_native
#check TensorBridge.kernel_real
#check TensorBridge.kernel_native
#check TensorBridge.empty_kernel
example {α : Type} (M : Algebra α) (D : Dot α) : (withDot M D).cast=M.cast := rfl
example {α : Type} (M : Algebra α) (D : Dot α) : (withDot M D).fp32Load=M.fp32Load := rfl
example (I : Instructions) (word : BitVec 32) : I.isa.bits word=word := rfl
#axiomsClean complete
#axiomsClean actual_address
#axiomsClean actual_read
#axiomsClean scale_read
#axiomsClean backing_exterior
#axiomsClean source_algorithm
#axiomsClean Correct.kernel_raw_correct
#axiomsClean FPEquiv.kernel_raw_native
#axiomsClean TensorBridge.kernel_real
#axiomsClean TensorBridge.kernel_native
#axiomsClean TensorBridge.empty_kernel
#specNonCircular mathematicalTarget avoiding [FP.Structural.exec,Whole.runPrograms,Whole.answer,Source.accumulated,Source.update]
#specNonCircular nativeReference avoiding [FP.Structural.exec,Whole.runPrograms,Whole.answer,Source.accumulated,Source.update]
#specNonCircular InputContract avoiding [FP.Structural.exec,Whole.runPrograms,Whole.answer]
end VeriTile.Bench.Repairs.AttentionFwdTriton2.KernelOnly
