import bench.repairs.tritonbench_g.cross_entropy1.Misc.HostFPEquiv

open Lean Elab Command in
run_meta do
  let env ← getEnv
  for name in env.header.moduleNames do
    if name.toString == "bench.repairs.tritonbench_g.cross_entropy1.Correct" then
      throwError "REVISION005_FLOATING_ROOT_IMPORTS_CORRECT"
  logInfo m!"REVISION005_FLOATING_ROOT_INDEPENDENT {env.header.moduleNames.size}"
