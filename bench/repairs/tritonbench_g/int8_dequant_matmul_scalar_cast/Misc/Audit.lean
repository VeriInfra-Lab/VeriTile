import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Misc.TensorBridge
import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Correct
import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Misc.NativeCorrect
import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.SourceBinding
import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.PreparedZero

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast

#axiomsClean TypedSource.typed_kernel_preserved
#axiomsClean typedJIT_preserved
#axiomsClean prepared_bounds
#axiomsClean runtime_cast_exact
#axiomsClean constexpr_one_cast
#axiomsClean zeroed_read
#axiomsClean zero_hook_prepares
#axiomsClean Quote.run_append
#axiomsClean Quote.source_decomposition
#axiomsClean Quote.preludeExecution
#axiomsClean Quote.preludeContext
#axiomsClean Quote.bodyExecution
#axiomsClean Quote.bodyContext
#axiomsClean Quote.rangeExecution
#axiomsClean Quote.epilogueExecution
#axiomsClean Quote.prelude_pm
#axiomsClean Quote.prelude_pn
#axiomsClean Quote.prelude_rm
#axiomsClean Quote.prelude_rn
#axiomsClean Quote.prelude_rk
#axiomsClean Quote.prelude_a
#axiomsClean Quote.prelude_b
#axiomsClean Quote.factorX
#axiomsClean Quote.factorW
#axiomsClean Quote.source_a
#axiomsClean Quote.source_b
#axiomsClean Quote.source_update
#axiomsClean Quote.accumulation_from
#axiomsClean Quote.source_accumulation
#axiomsClean Quote.epilogue_value
#axiomsClean Quote.epilogue_memory
#axiomsClean Quote.fullExecution_memory
#axiomsClean Quote.epilogue_descriptors
#axiomsClean Quote.fullResult
#axiomsClean Quote.Result.replay_eq
#axiomsClean Quote.source_value
#axiomsClean Quote.integer_plan_real
#axiomsClean Quote.integer_accumulation_real
#axiomsClean Quote.integer_partial_real
#axiomsClean Quote.physical_active
#axiomsClean Quote.sourceCommit_eq
#axiomsClean Quote.sourceGrid_eq
#axiomsClean physical_covers
#axiomsClean physical_unique
#axiomsClean execution_eq
#axiomsClean output_word
#axiomsClean frame
#axiomsClean native_execution
#axiomsClean empty_execution
#axiomsClean TensorBridge.correct
#axiomsClean TensorBridge.fp_equiv

#specNonCircular target avoiding [Quote.execute,Quote.sourceGrid,execute,program,TensorBridge.correct,TensorBridge.fp_equiv]
#specNonCircular Direct.logicalInput avoiding [Quote.execute,Quote.sourceGrid,execute,program]
#specNonCircular Direct.logicalWordReference avoiding [Quote.execute,Quote.sourceGrid,execute,program]
#specNonCircular InputContract avoiding [Quote.execute,Quote.sourceGrid,execute,program,target,Direct.logicalWordReference]
#specNonCircular Prepared avoiding [Quote.execute,Quote.sourceGrid,execute,program,target,Direct.logicalWordReference]
#specNonCircular Zeroed avoiding [Quote.execute,Quote.sourceGrid,execute,program,target,Direct.logicalWordReference]
#specNonCircular DotPlan avoiding [Quote.execute,Quote.sourceGrid,execute,program,target,Direct.logicalWordReference]
#specNonCircular FixedISAProfile avoiding [Quote.execute,Quote.sourceGrid,execute,program,target,Direct.logicalWordReference]
#specNonCircular LocalEventABI avoiding [Quote.execute,Quote.sourceGrid,execute,program,target,Direct.logicalWordReference]

end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast
