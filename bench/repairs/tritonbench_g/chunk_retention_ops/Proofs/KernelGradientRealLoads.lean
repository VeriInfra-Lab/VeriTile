import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelRealModel
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelGradientModel
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelReal
open VeriTile Triton FP.Structural
open KernelBackward (PID)
open KernelGradient
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

def history (c : Shape) (pid : PID) (m : Memory ℝ) : Tile ℝ :=
  fun e j=>(m "h" (pid.z*(c.chunks*c.K*c.V)+pid.y*c.K*c.V+e*c.V+j)).read .real
def dhistory (c : Shape) (pid : PID) (m : Memory ℝ) : Tile ℝ :=
  fun e j=>(m "dhistory" (pid.z*(c.chunks*c.K*c.V)+pid.y*c.K*c.V+e*c.V+j)).read .real

theorem grad_query_load (c : Shape) (pid : PID) (m : Memory ℝ) (seed terminal : Bool)
    (i : TileIndex [c.BK,64]) :
    gradQuery algebra (parameters c) pid m i=
      qAt realOps c (inputs c m seed terminal) pid.z (pid.y*64+i.2.1.val) (pid.x*c.BK+i.1.val) := by
  simp [gradQuery,VeriTile.Bench.Repairs.ChunkLinearAttn.load,VeriTile.Bench.Repairs.ChunkLinearAttn.rawLoad,
    VeriTile.Bench.Repairs.ChunkLinearAttn.zero,parameters,algebra,qAt,masked,realOps,inputs,
    and_comm,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
theorem grad_key_load (c : Shape) (pid : PID) (m : Memory ℝ) (seed terminal : Bool)
    (i : TileIndex [64,c.BK]) :
    gradKey algebra (parameters c) pid m i=
      kAt realOps c (inputs c m seed terminal) pid.z (pid.y*64+i.1.val) (pid.x*c.BK+i.2.1.val) := by
  simp [gradKey,VeriTile.Bench.Repairs.ChunkLinearAttn.load,VeriTile.Bench.Repairs.ChunkLinearAttn.rawLoad,
    VeriTile.Bench.Repairs.ChunkLinearAttn.zero,parameters,algebra,kAt,masked,realOps,inputs]
theorem grad_value_load (c : Shape) (pid : PID) (m : Memory ℝ) (seed terminal : Bool)
    (b : Nat) (i : TileIndex [64,c.BV]) :
    valueTile algebra (parameters c) pid m "v" b i=
      vAt realOps c (inputs c m seed terminal) pid.z (pid.y*64+i.1.val) (b*c.BV+i.2.1.val) := by
  simp [valueTile,VeriTile.Bench.Repairs.ChunkLinearAttn.load,VeriTile.Bench.Repairs.ChunkLinearAttn.rawLoad,
    VeriTile.Bench.Repairs.ChunkLinearAttn.zero,parameters,algebra,vAt,masked,realOps,inputs]
theorem grad_incoming_load (c : Shape) (pid : PID) (m : Memory ℝ) (seed terminal : Bool)
    (b : Nat) (i : TileIndex [64,c.BV]) :
    valueTile algebra (parameters c) pid m "incoming" b i=
      gAt realOps c (inputs c m seed terminal) pid.z (pid.y*64+i.1.val) (b*c.BV+i.2.1.val) := by
  simp [valueTile,VeriTile.Bench.Repairs.ChunkLinearAttn.load,VeriTile.Bench.Repairs.ChunkLinearAttn.rawLoad,
    VeriTile.Bench.Repairs.ChunkLinearAttn.zero,parameters,algebra,gAt,masked,realOps,inputs]
theorem grad_history_load (c : Shape) (pid : PID) (m : Memory ℝ)
    (b : Nat) (i : TileIndex [c.BV,c.BK]) :
    gradHistory algebra (parameters c) pid m b i=
      masked realOps c.K c.V (history c pid m) (pid.x*c.BK+i.2.1.val) (b*c.BV+i.1.val) := by
  simp [gradHistory,VeriTile.Bench.Repairs.ChunkLinearAttn.load,VeriTile.Bench.Repairs.ChunkLinearAttn.rawLoad,
    VeriTile.Bench.Repairs.ChunkLinearAttn.zero,parameters,algebra,history,masked,realOps,and_comm,
    Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
theorem grad_dhistory_load (c : Shape) (pid : PID) (m : Memory ℝ)
    (b : Nat) (i : TileIndex [c.BK,c.BV]) :
    gradHistoryGradient algebra (parameters c) pid m b i=
      masked realOps c.K c.V (dhistory c pid m) (pid.x*c.BK+i.1.val) (b*c.BV+i.2.1.val) := by
  simp [gradHistoryGradient,VeriTile.Bench.Repairs.ChunkLinearAttn.load,VeriTile.Bench.Repairs.ChunkLinearAttn.rawLoad,
    VeriTile.Bench.Repairs.ChunkLinearAttn.zero,parameters,algebra,dhistory,masked,realOps]

theorem key_decay_eq (c : Shape) (pid : PID) (i : Nat) :
    keyDecay algebra (parameters c) pid i=decay realOps c pid.z ((c.length pid.y:Int)-i-1) := by
  unfold keyDecay
  rw [logRate_eq]
  simp [VeriTile.Bench.Repairs.ChunkLinearAttn.mul,algebra,parameters,decay,realOps,Shape.length]
theorem query_decay_eq (c : Shape) (pid : PID) (scale : ℝ) (i : Nat) :
    queryDecay algebra (parameters c) pid scale i=decay realOps c pid.z ((i:Int)+1)*scale := by
  unfold queryDecay
  rw [weight_eq]
  rfl
theorem score_decay_eq (c : Shape) (pid : PID) (scale : ℝ) (i : TileIndex [64,64]) :
    scoreDecay algebra (parameters c) pid scale i=causalWeight realOps c pid.z i.1.val i.2.1.val*scale := by
  unfold scoreDecay
  rw [logRate_eq]
  simp [VeriTile.Bench.Repairs.ChunkLinearAttn.mul,VeriTile.Bench.Repairs.ChunkLinearAttn.scalar,
    VeriTile.Bench.Repairs.ChunkLinearAttn.zero,algebra,causalWeight,decay,realOps]
  norm_cast

#axiomsClean grad_query_load
#axiomsClean grad_history_load
#axiomsClean key_decay_eq
#axiomsClean score_decay_eq
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelReal
