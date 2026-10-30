import bench.tritonbench_g.matmul_persistent_triton.Model

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
open VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

structure Registers {α : Type} (p : Parameters) (pid : Nat) (s : State α) : Prop where
  start : s.regs .int [] "start_pid" = some (fun _ => (pid : Int))
  rows : s.regs .nat [] "num_pid_m" = some (fun _ => p.problem.rows)
  columns : s.regs .nat [] "num_pid_n" = some (fun _ => p.problem.columns)
  count : s.regs .nat [] "k_tiles" = some (fun _ => p.problem.kTiles)
  work : s.regs .nat [] "tiles_per_SM" = some (fun _ => p.problem.work pid)
  group : s.regs .nat [] "active_group_m" = some (fun _ => p.problem.group)
  span : s.regs .nat [] "num_pid_in_group" = some (fun _ => p.problem.group*p.problem.columns)
  lanesM : s.regs .int [128] "lanes_m" = some (fun i => (i.1.val : Int))
  lanesN : s.regs .int [256] "lanes_n" = some (fun i => (i.1.val : Int))
  lanesK : s.regs .int [p.problem.profile.blockK] "offs_k_for_mask" = some (fun i => (i.1.val : Int))

end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
