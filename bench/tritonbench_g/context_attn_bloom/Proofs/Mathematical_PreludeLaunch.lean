import VeriTile.Triton.Semantics.Integer
import bench.tritonbench_g.context_attn_bloom.Proofs.Mathematical_PreludeProgram
import bench.tritonbench_g.context_attn_bloom.Proofs.Mathematical_RealSetup
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.PreludeLaunch
open VeriTile.Triton Math Program RealSetup PreludeProgram
noncomputable section
attribute [local simp 2000] VeriTile.Triton.eval_truncDiv VeriTile.Triton.eval_truncMod
set_option maxHeartbeats 500000
set_option maxRecDepth 2000
set_option linter.unusedSimpArgs false

private theorem nonnegative_division (a b : Nat) :
    (a : Int).tdiv (b : Int) = ((a / b : Nat) : Int) := by
  rw [Int.tdiv_eq_ediv_of_nonneg (Int.natCast_nonneg a)]
  rfl
attribute [local simp 2000] nonnegative_division

def launched (p : Layout) (u : BlockState) : BlockState :=
  let l := launch u
  let u := u.setReg "cur_batch" .int [] (Tile.scalar l.batch)
  let u := u.setReg "cur_head" .int [] (Tile.scalar l.head)
  let u := u.setReg "start_m" .int [] (Tile.scalar l.block)
  u.setReg "cur_kv_head" .int [] (Tile.scalar (l.head/p.kvGroup))

theorem executes (p : Layout) (u : BlockState) :
    stepStmts (launchCode p) u = some (launched p u) := by
  simp [launchCode,launched,launch,n,z,stepStmts,stepStmt,evalOp.eq_def,
    tile_elementwise,Tile.scalar,Broadcast.leftIndex,Broadcast.rightIndex,
    Option.bind_eq_bind,Option.bind,-Tile.scalar_eta]

theorem memory (p : Layout) (u : BlockState) : (launched p u).mem = u.mem := by
  funext region off
  simp only [launched,BlockState.setReg_mem]
theorem pids (p : Layout) (u : BlockState) : (launched p u).pids = u.pids := by
  simp only [launched,BlockState.setReg_pids]
theorem batch (p : Layout) (u : BlockState) :
    (launched p u).regs .int [] "cur_batch" = some (Tile.scalar (u.pids 0)) := by
  simp [launched,launch]
theorem head (p : Layout) (u : BlockState) :
    (launched p u).regs .int [] "cur_head" = some (Tile.scalar (u.pids 1)) := by
  simp [launched,launch]
theorem block (p : Layout) (u : BlockState) :
    (launched p u).regs .int [] "start_m" = some (Tile.scalar (u.pids 2)) := by
  simp [launched,launch]
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.PreludeLaunch
