import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.DSL
import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Correct
import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Misc.NativeCorrect
import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Proofs.Frames
import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Proofs.Safety
import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Proofs.S1
import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Proofs.RawTensor
import bench.repairs.tritonbench_g.matmul_dequant_int4.Correct
import bench.repairs.tritonbench_g.matmul_dequant_int4.FPEquiv

/-! Public whole-task entry point. The two source quotations and both complete
kernel families have independent Real/native and actual Tensor/frame proofs.
Historical ordinary Torch quantizer/mm and host proofs remain imported under
their original namespaces; they are not new kernel admission premises. -/
#guard_msgs (drop info) in
#auditModuleAxioms
