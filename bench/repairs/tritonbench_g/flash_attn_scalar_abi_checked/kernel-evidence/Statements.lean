import bench.repairs.tritonbench_g.flash_attn_scalar_abi_checked.Correct
import bench.repairs.tritonbench_g.flash_attn_scalar_abi_checked.FPEquiv

open VeriTile.Bench.Repairs
#print FlashAttnScalarABIChecked.shared_source
#print FlashAttnScalarABIChecked.check_iff
#print FlashAttnScalarABIChecked.whole_real
#print FlashAttnScalarABIChecked.whole_native
#axiomsClean FlashAttnScalarABIChecked.shared_source
#axiomsClean FlashAttnScalarABIChecked.check_iff
#axiomsClean FlashAttnScalarABIChecked.whole_real
#axiomsClean FlashAttnScalarABIChecked.whole_native
#axiomsClean FlashAttn.ActualSource.source_memory
#axiomsClean FlashAttnScalarABIChecked.input_tensor
#axiomsClean FlashAttn.KernelTensor.logical_read
#axiomsClean FlashAttn.KernelNative.output_owner
#axiomsClean FlashAttn.KernelNative.constexpr_one_fold
#specNonCircular FlashAttn.KernelTensor.nativeTarget avoiding [VeriTile.Triton.FP.Structural.exec,VeriTile.Triton.FP.Structural.run,FlashAttn.ActualSource.runGrid,FlashAttn.KernelTensor.run]
#specNonCircular FlashAttn.Spec.output avoiding [VeriTile.Triton.FP.Structural.exec,VeriTile.Triton.FP.Structural.run,FlashAttn.ActualSource.runGrid,FlashAttn.ActualSource.referenceGrid]
#specNonCircular FlashAttn.Spec.lse avoiding [VeriTile.Triton.FP.Structural.exec,VeriTile.Triton.FP.Structural.run,FlashAttn.ActualSource.runGrid,FlashAttn.ActualSource.referenceGrid]
