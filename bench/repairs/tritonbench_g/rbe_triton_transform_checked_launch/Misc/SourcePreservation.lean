import bench.repairs.tritonbench_g.rbe_triton_transform_checked_launch.Kernels
import bench.repairs.tritonbench_g.rbe_triton_transform.Proofs.SelectedExecution

namespace VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch
open VeriTile.Bench.Repairs

/-- Complete source identity: public regions, source precision and every statement.
Only the already accepted index/store adapter has been written out explicitly. -/
theorem source_is_frozen (p : RbeTritonTransform.Source.Parameters) :
    source p = RbeTritonTransform.Selected.source p := by rfl

#axiomsClean source_is_frozen
end VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch
