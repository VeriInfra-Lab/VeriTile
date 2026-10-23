import bench.repairs.tritonbench_g.cross_entropy1.Misc.HostCorrect
import bench.repairs.tritonbench_g.cross_entropy1.Misc.HostFPEquiv
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.TorchSchedule

open Lean Elab Command in
run_meta do
  let env ← getEnv
  for name in env.header.moduleNames do
    logInfo m!"REVISION003_IMPORTED_MODULE {name}"
