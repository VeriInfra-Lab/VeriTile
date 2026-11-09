import bench.tritonbench_g.matmul_persistent_triton.Proofs.Support.Model

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
  rows : s.regs .int [] "num_pid_m" = some (fun _ => (p.problem.rows : Int))
  columns : s.regs .int [] "num_pid_n" = some (fun _ => (p.problem.columns : Int))
  count : s.regs .int [] "k_tiles" = some (fun _ => (p.problem.kTiles : Int))
  work : s.regs .int [] "tiles_per_SM" = some (fun _ => (p.problem.work pid : Int))
  group : s.regs .int [] "active_group_m" = some (fun _ => (p.problem.group : Int))
  span : s.regs .int [] "num_pid_in_group" = some (fun _ => (p.problem.group*p.problem.columns : Int))
  lanesM : s.regs .int [128] "lanes_m" = some (fun i => (i.1.val : Int))
  lanesN : s.regs .int [256] "lanes_n" = some (fun i => (i.1.val : Int))
  lanesK : s.regs .int [p.problem.profile.blockK] "offs_k_for_mask" = some (fun i => (i.1.val : Int))

  args : Arguments p s

end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
