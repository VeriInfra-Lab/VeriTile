import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Correct
import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.FPEquiv
import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Misc.TensorBridge
namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked
open VeriTile Triton
set_option maxHeartbeats 10000000

#axiomsClean correct
#axiomsClean fp_equiv
#axiomsClean TensorBridge.correct
#axiomsClean TensorBridge.fp_equiv
#axiomsClean Contract.python_checks_iff
#axiomsClean Contract.selected_valid
#axiomsClean Signed.Program.complete_code
#axiomsClean Signed.Program.complete_active
#axiomsClean Signed.Program.signed_length
#axiomsClean Signed.Setup.preguard
#axiomsClean Signed.Body.iteration
#axiomsClean Signed.Execution.source_executes
#axiomsClean Signed.Grid.complete
#axiomsClean Real.source_result
#axiomsClean Native.source_native
#axiomsClean Native.instructions_eq
#axiomsClean native_reads
#axiomsClean decode_encode
#axiomsClean encode_decode
#axiomsClean pointer_address
#axiomsClean raw_pointer
#axiomsClean outputIndex_unique
#axiomsClean address_unique
#axiomsClean nativeStores_word
#axiomsClean nativeStores_logical
#axiomsClean nativeStores_untouched
#axiomsClean mathStores_logical
#axiomsClean mathStores_untouched
#axiomsClean writeback_frame
#axiomsClean writeback_inputs
#axiomsClean static_tile
#axiomsClean source_shapes
#axiomsClean signed_cdiv
#axiomsClean all_formed
#axiomsClean Direct.Bounds.signed_offsets

#specNonCircular Admitted avoiding [FP.Structural.exec,Real.execute,Native.execute,tensorTarget,Native.reference,mathStores,nativeStores]
#specNonCircular Request avoiding [FP.Structural.exec,Real.execute,Native.execute,tensorTarget,Native.reference,mathStores,nativeStores]
#specNonCircular Contract.ByteSeparate avoiding [FP.Structural.exec,Real.execute,Native.execute,mathStores,nativeStores]
#specNonCircular tensorTarget avoiding [FP.Structural.exec,Real.execute,Real.formula,mathStores]
#specNonCircular Native.reference avoiding [FP.Structural.exec,Signed.Grid.run,Direct.FPGrid.run,Direct.FPGrid.fold,Native.execute,nativeStores]
#specNonCircular Native.LocalConformance avoiding [FP.Structural.exec,Signed.Grid.run,Native.execute,Native.Expr.eval,Native.reference,nativeStores]
#specNonCircular Native.inputs avoiding [Native.execute,Native.reference,nativeStores]
#specNonCircular mathStores avoiding [tensorTarget,Real.execute]
#specNonCircular nativeStores avoiding [Native.reference,Native.execute]

/- These audits traverse proof bodies. Auxiliary pure value/fold/address
lemmas may be reused, but neither public theorem depends on the normalized
source or its execution proofs. -/
#specNonCircular correct avoiding [Direct.Kernels.source,Direct.FPProgram.primarySource,Direct.FPGrid.run,Direct.FPGrid.executes,Direct.FPExecution.source_executes,VeriTile.Bench.Repairs.BgmvExpandSlice.Kernels.directSource]
#specNonCircular fp_equiv avoiding [Direct.Kernels.source,Direct.FPProgram.primarySource,Direct.FPGrid.run,Direct.FPGrid.executes,Direct.FPExecution.source_executes,VeriTile.Bench.Repairs.BgmvExpandSlice.Kernels.directSource]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked
