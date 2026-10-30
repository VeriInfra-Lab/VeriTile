import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardModel
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MemoryHelpers
namespace VeriTile.Bench.Repairs.LayerNormOps.Backward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits (fold_writes_mem fold_writes_regs)
open Forward (fold_writes_undef vector_bop region_name_eq)
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option pp.maxSteps 1200
set_option linter.unusedSimpArgs false

structure Ready {α : Type} (p : BackwardParameters) (s t : State α) (row : Nat) (a : Acc α p) : Prop where
  mem : t.mem = a.state.mem
  pid : t.regs .nat [] "row_block_id" = some (fun _ => s.pids 0)
  first : t.regs .nat [] "row_start" = some (fun _ => start p s)
  last : t.regs .nat [] "row_end" = some (fun _ => stop p s)
  cols : t.regs .nat [p.block] "cols" = some (fun i => i.1.val)
  mask : t.regs .bool [p.block] "mask" = some (fun i => decide (i.1.val < p.columns))
  dw : t.regs .real [p.block] "dw" = some a.dw
  db : p.hasBias = Bool.true → t.regs .real [p.block] "db" = some a.db
  weight : t.regs .real [p.block] "w" = some a.weight
  bias : p.recompute = Bool.true → p.hasBias = Bool.true → t.regs .real [p.block] "b" = some a.bias
  xp : t.regs .ptr [] "X" = some (fun _ => (p.x,row*p.xStride))
  yp : p.recompute = Bool.true → t.regs .ptr [] "Y" = some (fun _ => (p.y,row*p.yStride))
  dxp : t.regs .ptr [] "DX" = some (fun _ => (p.dx,row*p.dxStride))
  dyp : t.regs .ptr [] "DY" = some (fun _ => (p.dy,row*p.dyStride))
  dres : p.hasDres = Bool.true → t.regs .ptr [] "DRESIDUAL" = some (fun _ => (p.dres,row*p.dresStride))
  dresIn : p.storeDres = Bool.true → t.regs .ptr [] "DRESIDUAL_IN" = some (fun _ => (p.dresIn,row*p.dresInStride))

end
end VeriTile.Bench.Repairs.LayerNormOps.Backward
