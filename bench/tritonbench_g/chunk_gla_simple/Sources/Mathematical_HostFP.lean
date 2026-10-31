import bench.tritonbench_g.chunk_gla_simple.Sources.Mathematical_FP
import bench.tritonbench_g.chunk_gla_simple.Sources.Mathematical_HostGeometry
import VeriTile.Triton.Float.ProfiledSoundness

/-! FP host composition imports no Real execution proof. Every actual grid
instance receives the source PIDs and the reached finite-total/scale guard.
Equality includes the entire optional result, hence successful return values,
all registers and tagged memory, as well as matching failure behavior. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple.HostFP
open VeriTile Triton FP.Structural FP.Guarded FP.ProfiledRewrite Host
open scoped VeriTile.Spec
noncomputable section
set_option maxHeartbeats 2000000

def atProgram {α : Type} (c : Shape) (s : State α) (idx : GridIndex (grid c)) : State α :=
  { s with pids := idx.toPids, numPids := (grid c).toNumPids }

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaSimple.HostFP
