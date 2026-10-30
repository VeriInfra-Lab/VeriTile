import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Correct
import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.FPEquiv
import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Misc.NativeCorrect
import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Misc.TensorBridge

open VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked
#print inputChecks
#print arange_bounds
#print Source.execution
#print Source.execution_memory
#print Source.program_memory
#print Source.complete
#print Source.complete_real
#print NativeSource.source_selected
#print NativeSource.complete_bytes
#print TensorObservation.Binding
#print TensorObservation.logical_read
#print Correct.fused_rotary_real
#print FPEquiv.fused_rotary_native
#axiomsClean Correct.fused_rotary_real
#axiomsClean FPEquiv.fused_rotary_native
#specNonCircular VeriTile.Bench.Repairs.FusedRotaryEmbedding.formula avoiding [Source.kernel,Source.grid,Source.invocation]
#specNonCircular VeriTile.Bench.Repairs.FusedRotaryEmbedding.wordFormula avoiding [Source.kernel,Source.grid,Source.invocation]
#specNonCircular TensorObservation.Binding avoiding [Source.kernel,Source.grid,Source.invocation]
#specNonCircular FPEquiv.fused_rotary_native avoiding [VeriTile.Bench.Repairs.FusedRotaryEmbedding.realOps,VeriTile.Bench.Repairs.FusedRotaryEmbedding.formula]

#print FPEquiv.kernel_equivalent
#axiomsClean FPEquiv.kernel_equivalent
#axiomsClean Source.source_safe
