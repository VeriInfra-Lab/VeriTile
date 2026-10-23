import bench.repairs.tritonbench_g.cross_entropy1.Misc.HostCorrect
import bench.repairs.tritonbench_g.cross_entropy1.Misc.BindingRejections
import bench.repairs.tritonbench_g.cross_entropy1.Misc.FloatingDependencyAudit
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.TorchSchedule

open Lean Elab Command in
run_meta do
  let env ← getEnv
  for name in env.header.moduleNames do
    logInfo m!"REVISION004_IMPORTED_MODULE {name}"
