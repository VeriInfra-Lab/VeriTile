import bench.repairs.tritonbench_g.chunk_gated_attention.KernelOnly.Correct
import bench.repairs.tritonbench_g.chunk_gated_attention.KernelOnly.Misc.NativeCorrect
import bench.repairs.tritonbench_g.chunk_gated_attention.KernelOnly.FPEquiv
import bench.repairs.tritonbench_g.chunk_gated_attention.KernelOnly.Misc.TensorBridge
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.RawReal
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.RawFP

/-! Reader-entry and retained-scope audits live at the task root. Operational
helpers never import Correct, FPEquiv or TensorBridge. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention
open VeriTile Triton
open KernelOnly

#axiomsClean KernelOnly.inner_correct
#axiomsClean KernelOnly.prefix_correct
#axiomsClean KernelOnly.inner_fp_equiv
#axiomsClean KernelOnly.prefix_fp_equiv
#axiomsClean RawReal.inner_correct
#axiomsClean RawReal.prefix_correct
#axiomsClean RawFP.inner_fp_equiv
#axiomsClean RawFP.prefix_fp_equiv
#axiomsClean RealInterface.selected_projection
#axiomsClean RealInterface.inner_implements
#axiomsClean RealInterface.prefix_implements
#axiomsClean NativeSafety.inner_safe
#axiomsClean NativeSafety.prefix_safe
#axiomsClean NativeInterface.inner_binding
#axiomsClean NativeInterface.prefix_binding
#axiomsClean NativeInterface.inner_io_safe
#axiomsClean NativeInterface.prefix_io_safe
#axiomsClean NativeInterface.inner_implements
#axiomsClean NativeInterface.prefix_implements
#axiomsClean NativeInterface.inner_schedule_original
#axiomsClean NativeInterface.prefix_schedule_original
#axiomsClean TensorBridge.inner_correct
#axiomsClean TensorBridge.prefix_correct
#axiomsClean TensorBridge.inner_fp_equiv
#axiomsClean TensorBridge.prefix_fp_equiv
#specNonCircular RealInterface.innerFormula avoiding [VeriTile.Triton.exec,AlgorithmExecution]
#specNonCircular RealInterface.prefixFormula avoiding [VeriTile.Triton.exec,AlgorithmExecution]
#specNonCircular NativeInterface.innerNative avoiding [FP.Structural.exec,FP.Structural.run,
  FP.Structural.runSchedule,HostFP.batches]
#specNonCircular NativeInterface.prefixNative avoiding [FP.Structural.exec,FP.Structural.run,
  FP.Structural.runSchedule,HostFP.batches]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkGatedAttention
