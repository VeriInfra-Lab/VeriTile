import bench.tritonbench_g.quantize_global.Proofs.FiniteRetainedBridge
import bench.tritonbench_g.quantize_global.FullSourceCorrect

open Lean Elab Command in
elab "#quantFiniteSupport" : command => do
  let env ← getEnv
  for entry in env.header.modules do
    liftIO <| IO.println s!"dependency {entry.module}"

#quantFiniteSupport

#guard_msgs (drop info) in
#auditModuleAxioms
