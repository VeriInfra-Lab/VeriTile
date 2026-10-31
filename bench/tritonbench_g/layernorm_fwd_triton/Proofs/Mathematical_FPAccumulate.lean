import bench.tritonbench_g.layernorm_fwd_triton.Proofs.Mathematical_FPModel
namespace VeriTile.Bench.TritonBenchG.LayernormFwdTriton.FPStages
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

structure MeanReady {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s t : State α)
    (a : TileIndex [p.block] → α) : Prop where
  mem : t.mem = s.mem
  pids : t.pids = s.pids
  undef : t.undef = s.undef
  x : t.regs .ptr [] "X" = some (fun _ => (p.x,xBase p s))
  y : t.regs .ptr [] "Y" = some (fun _ => (p.y,yBase p s))
  w : t.regs .ptr [] "W" = some (fun _ => (p.weight,wBase p s))
  accumulator : t.regs .real [p.block] "_mean" = some a

structure VarReady {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s t : State α)
    (a : TileIndex [p.block] → α) : Prop where
  mem : t.mem = s.mem
  pids : t.pids = s.pids
  undef : t.undef = s.undef
  x : t.regs .ptr [] "X" = some (fun _ => (p.x,xBase p s))
  y : t.regs .ptr [] "Y" = some (fun _ => (p.y,yBase p s))
  w : t.regs .ptr [] "W" = some (fun _ => (p.weight,wBase p s))
  accumulator : t.regs .real [p.block] "_var" = some a
  mean : t.regs .real [] "mean" = some (fun _ => meanValue M p s)

end
end VeriTile.Bench.TritonBenchG.LayernormFwdTriton.FPStages
