import bench.repairs.tritonbench_g.context_attn_fwd.Kernels
import bench.repairs.tritonbench_g.context_attn_fwd.Contract
import bench.repairs.tritonbench_g.context_attn_fwd.Correct
import bench.repairs.tritonbench_g.context_attn_fwd.FPEquiv

open VeriTile.Bench.Repairs.ContextAttnFwd
#print PublicKernel.shared_source
#print PublicKernel.correct_real
#print PublicKernel.correct_fp32
#print PublicKernel.input_tensor
#print KernelOnly.nativeTarget
#print KernelOnly.mathematical
#print NativeGraph.ScalarOps
#print NativeSM90.plan
#print KernelScale.Bound
#axiomsClean PublicKernel.shared_source
#axiomsClean PublicKernel.correct_real
#axiomsClean PublicKernel.correct_fp32
#axiomsClean PublicKernel.input_tensor
#axiomsClean KernelProfile.route_payload_independent
#axiomsClean KernelProfile.operand_order_payload_independent
#axiomsClean KernelProfile.accumulate_payload_independent
#axiomsClean KernelProfile.pv_seed_always_live
#specNonCircular KernelOnly.nativeTarget avoiding [VeriTile.Triton.FP.Structural.exec,
  VeriTile.Triton.FP.Structural.run,FPSuccess.runGrid,kernel,KernelOnly.run,Returned.invoke]
#specNonCircular KernelOnly.mathematical avoiding [VeriTile.Triton.exec,
  VeriTile.Triton.FP.Structural.exec,kernel,KernelOnly.run,Returned.invoke]
