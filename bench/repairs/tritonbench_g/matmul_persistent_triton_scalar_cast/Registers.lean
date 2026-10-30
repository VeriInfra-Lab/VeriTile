import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Model

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
  rows : s.regs .int [] "num_pid_m" = some (fun _ => ((p.problem.rows : Nat) : Int))
  columns : s.regs .int [] "num_pid_n" = some (fun _ => ((p.problem.columns : Nat) : Int))
  count : s.regs .int [] "k_tiles" = some (fun _ => ((p.problem.kTiles : Nat) : Int))
  work : s.regs .int [] "tiles_per_SM" = some (fun _ => ((p.problem.work pid : Nat) : Int))
  group : s.regs .int [] "active_group_m" = some (fun _ => ((p.problem.group : Nat) : Int))
  span : s.regs .int [] "num_pid_in_group" = some (fun _ => ((p.problem.group*p.problem.columns : Nat) : Int))
  lanesM : s.regs .int [128] "lanes_m" = some (fun i => (i.1.val : Int))
  lanesN : s.regs .int [256] "lanes_n" = some (fun i => (i.1.val : Int))
  lanesK : s.regs .int [p.problem.profile.blockK] "offs_k_for_mask" = some (fun i => (i.1.val : Int))
  dimM : s.regs .int [] "M" = some (fun _ => (p.problem.M : Int))
  dimN : s.regs .int [] "N" = some (fun _ => (p.problem.N : Int))
  dimK : s.regs .int [] "K" = some (fun _ => (p.problem.K : Int))
  strideAM : s.regs .int [] "stride_am" = some (fun _ => (p.am : Int))
  strideAK : s.regs .int [] "stride_ak" = some (fun _ => (p.ak : Int))
  strideBK : s.regs .int [] "stride_bk" = some (fun _ => (p.bk : Int))
  strideBN : s.regs .int [] "stride_bn" = some (fun _ => (p.bn : Int))
  strideCM : s.regs .int [] "stride_cm" = some (fun _ => (p.cm : Int))
  strideCN : s.regs .int [] "stride_cn" = some (fun _ => (p.cn : Int))

end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
