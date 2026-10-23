import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Correct
import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Misc.NativeCorrect
import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Proofs.TensorResults

/-! Reader entry for the retained actual-Tensor, physical-output and byte-frame
results. Their original names and full statements are defined in the reusable
`Proofs/TensorResults.lean` foundation, without importing this reader entry. -/

#guard_msgs (drop info) in
#auditModuleAxioms
