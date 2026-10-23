import bench.repairs.tritonbench_g.context_attn_bloom.Whole001.Correct
import bench.repairs.tritonbench_g.context_attn_bloom.Whole001.FPEquiv

namespace ContextAttnBloomWhole001Reader
open Lean Elab Command
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ContextAttnBloomWhole001
#check @Public.host_real
#check @Public.host_typed
#check @Public.host_native
#check @Public.host_native_coverage
#print HostRun.LayoutDomain
#print Host.Admissible
#print Host.paddedDim
#print Host.scale
#print Host.grid
#print Preparation.LocalAPI
#print Preparation.stages
#print ABI.IndexABI
#print ABI.SourceExact
#print ABI.programRoots
#print Native.WordISA
#print Native.Plan
#print Native.lower
#print RealStream.run
#print RealStream.flatRun
#print FPSuccess.runGrid
#axiomsClean ABI.word_exact
#axiomsClean ABI.expression_exact
#axiomsClean ABI.source_exact
#axiomsClean ABI.excludes_overflowing_head_stride
#axiomsClean Preparation.fold_bound
#axiomsClean Preparation.prepared_inputs
#axiomsClean Preparation.output_frame
#axiomsClean Preparation.caller_frame
#axiomsClean Preparation.undef_frame
#axiomsClean Host.unique_writer
#axiomsClean Host.covers
#axiomsClean RealStream.program
#axiomsClean RealStream.complete
#axiomsClean FPSuccess.prelude_success
#axiomsClean FPSuccess.body_success
#axiomsClean FPSuccess.program_success
#axiomsClean FPSuccess.grid_success_frame
#axiomsClean Native.dot_feature_covered
#axiomsClean Native.dot_feature_unique
#axiomsClean Native.reduction_covered
#axiomsClean Native.reduction_nodup
#axiomsClean Native.transport
#axiomsClean Native.lowered_transport
#axiomsClean Public.expected_eq
#axiomsClean Public.host_real
#axiomsClean Public.host_typed
#axiomsClean Public.host_native
#axiomsClean Public.host_native_coverage
#auditStmt Public.host_real
#auditStmt Public.host_typed
#auditStmt Public.host_native
#specNonCircular Public.expected avoiding [FP.Structural.exec,RealStream.run,FPSuccess.runGrid,HostRun.real,HostRun.native]
#specNonCircular Native.lower avoiding [FP.Structural.exec,RealStream.run,FPSuccess.runGrid,HostRun.real,HostRun.native]
run_cmd do
  let env ← getEnv
  let names := env.header.modules.map (fun m => Json.str m.module.toString)
  logInfo m!"BLOOM_READER_MODULES_JSON {(Json.arr names).compress}"
run_cmd do
  let names := #[`Public.host_real,`Public.host_typed,`Public.host_native,`Public.host_native_coverage]
  for short in names do
    let n := `VeriTile.Bench.Repairs.ContextAttnBloomWhole001 ++ short
    let ci ← liftCoreM <| getConstInfo n
    let txt ← liftTermElabM <| Lean.Meta.ppExpr ci.type
    logInfo m!"BLOOM_READER_STATEMENT_JSON {(Json.mkObj [("name",Json.str n.toString),("type",Json.str txt.pretty)]).compress}"
end ContextAttnBloomWhole001Reader
