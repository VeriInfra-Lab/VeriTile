import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Misc.Compatibility
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Misc.TensorBridge
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Correct
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.FPEquiv
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.KernelPrecision

namespace RmsnormFusedLlamaReader
open Lean Elab Command
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.RmsnormFusedLlama

#check @Correct.rmsnorm_kernel_real
#check @FPEquiv.rmsnorm_kernel_native
#check @TensorBridge.kernel_correct
#check @TensorBridge.kernel_native_equivalence
#print KernelAPI.Binding
#print EpsilonABI.Binding
#print KernelAPI.mathematical
#print RuntimeReference.expression
#print Native.Instructions
#print Native.Shape
#print Native.Schedule
#print Native.Reductions
#print Native.Plan
#print Native.eval
#print Native.reduction

#axiomsClean TensorBridge.kernel_correct
#axiomsClean TensorBridge.kernel_native_equivalence
#axiomsClean RuntimeRow.complete
#axiomsClean RuntimeGrid.complete
#axiomsClean RuntimeGrid.graph_result
#axiomsClean RuntimeCandidate.typed
#axiomsClean RealGrid.complete
#axiomsClean Native.Plan.lane_coverage
#axiomsClean Native.leaf_shape
#axiomsClean EpsilonABI.positive_underflow
#axiomsClean EpsilonABI.positive_overflow
#axiomsClean EpsilonABI.halfway_even
#axiomsClean KernelPrecision.runtime_epsilon
#axiomsClean KernelPrecision.runtime_memory
#axiomsClean KernelAPI.signed_address
#axiomsClean KernelPublic.backings_frame
#axiomsClean KernelPublic.empty_kernel

#auditStmt TensorBridge.kernel_correct
#auditStmt TensorBridge.kernel_native_equivalence
#specNonCircular KernelAPI.mathematical avoiding [kernel,Triton.exec,RealGrid.run]
#specNonCircular KernelAPI.Binding avoiding [FP.Structural.exec,RuntimeGrid.run,RuntimeSource.nativeTarget]
#specNonCircular EpsilonABI.Binding avoiding [FP.Structural.exec,RuntimeGrid.run,RuntimeSource.nativeTarget]
#specNonCircular RuntimeSource.nativeTarget avoiding [runtimeKernel,FP.Structural.exec,RuntimeGrid.run,RuntimeGrid.graph,RuntimeRow.value,KernelSource.entry]
#specNonCircular Native.eval avoiding [FP.Structural.exec,RuntimeGrid.run]
#specNonCircular Native.reduction avoiding [FP.Structural.exec,RuntimeGrid.run]

example : Native.Reductions := fun _ _ s ax _ _ _ => Native.Plan.seeded (TileShape.axisDim s ax)
example : Native.Schedule := { fusion:=fun _=>.split,swap:=fun _=>Bool.false }
example : Native.Schedule := { fusion:=fun _=>.left,swap:=fun _=>Bool.true }
example : Native.Schedule := {
  fusion := fun _=>.right
  swap := fun _=>Bool.false
  elideInitialZero := Bool.true
  eraseFp32Casts := Bool.true
  elideDivisionOne := Bool.true
  reciprocalSqrt := Bool.true }

run_cmd do
  let env ← getEnv
  let names := env.header.modules.map (fun m => Json.str m.module.toString)
  logInfo m!"LLAMA_READER_MODULES_JSON {(Json.arr names).compress}"

run_cmd do
  for short in #[`TensorBridge.kernel_correct,`TensorBridge.kernel_native_equivalence,
      `RuntimeCandidate.typed,`KernelAPI.signed_address,
      `KernelPrecision.runtime_epsilon,`KernelPrecision.runtime_memory] do
    let n := `VeriTile.Bench.Repairs.RmsnormFusedLlama ++ short
    let ci ← liftCoreM <| getConstInfo n
    let txt ← liftTermElabM <| Lean.Meta.ppExpr ci.type
    logInfo m!"LLAMA_READER_STATEMENT_JSON {(Json.mkObj [("name",Json.str n.toString),("type",Json.str txt.pretty)]).compress}"
end RmsnormFusedLlamaReader
