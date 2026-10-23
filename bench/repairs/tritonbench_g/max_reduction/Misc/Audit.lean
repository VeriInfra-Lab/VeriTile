import bench.repairs.tritonbench_g.max_reduction.Correct
import bench.repairs.tritonbench_g.max_reduction.FPEquiv
import bench.repairs.tritonbench_g.max_reduction.Misc.NativeCorrect
import bench.repairs.tritonbench_g.max_reduction.Misc.TensorBridge
import bench.repairs.tritonbench_g.max_reduction.Proofs.LegacyRealResults
import bench.repairs.tritonbench_g.max_reduction.Proofs.SourceEquivalence
import bench.repairs.tritonbench_g.max_reduction.Misc.ASTCorrect
import bench.repairs.tritonbench_g.max_reduction.Misc.ASTHost

/-! Keep every prior flat-memory, raw-state, full-grid and host application
available beside the source-safe public IO statements; these are reader imports. -/
#guard_msgs (drop info) in
#auditModuleAxioms
