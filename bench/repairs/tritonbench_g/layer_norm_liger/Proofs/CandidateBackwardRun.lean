import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardRun
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.CandidateBackwardSource
import bench.tritonbench_g.isfinite_kernel.Proofs.MemoryProjection
namespace VeriTile.Bench.Repairs.LayerNormLiger.Backward.Candidate
open VeriTile.Triton FP.Structural
open LayerNormLiger.Backward
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive withSum)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits (fold_writes_mem fold_writes_regs)
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

structure Ready {α : Type} (M : Algebra α) (p : Parameters) (s t : State α) (row : Nat) (a : Acc α p) : Prop where
  mem : t.mem = a.state.mem
  pid : t.regs .int [] "row_block_id" = some (fun _ => (s.pids 0 : Int))
  first : t.regs .int [] "row_start" = some (fun _ => (start p s : Int))
  last : t.regs .int [] "row_end" = some (fun _ => (stop p s : Int))
  cols : t.regs .int [p.block] "cols" = some (fun i => (i.1.val : Int))
  mask : t.regs .bool [p.block] "mask" = some (fun i => decide (i.1.val < p.columns))
  dw : t.regs .real [p.block] "dw_row" = some a.dw
  db : t.regs .real [p.block] "db_row" = some a.db
  xp : t.regs .ptr [] "X_ptr" = some (fun _ => (p.x,row*p.xStride))
  mp : t.regs .ptr [] "Mean_ptr" = some (fun _ => (p.mean,row))
  rp : t.regs .ptr [] "RSTD_ptr" = some (fun _ => (p.rstd,row))
  dxp : t.regs .ptr [] "DX_ptr" = some (fun _ => (p.dx,row*p.dxStride))
  denominator : t.regs .real [] "denominator" = some (fun _ => denom M p)
  dyp : t.regs .ptr [] "DY_ptr" = some (fun _ => (p.dy,row*p.dyStride))

  sx : t.regs .int [] "stride_x" = some (fun _ => (p.xStride : Int))
  sdx : t.regs .int [] "stride_dx" = some (fun _ => (p.dxStride : Int))
  sdw : t.regs .int [] "stride_dw" = some (fun _ => (p.dwStride : Int))
  sdb : t.regs .int [] "stride_db" = some (fun _ => (p.dbStride : Int))
  sdy : t.regs .int [] "stride_dy" = some (fun _ => (p.dyStride : Int))
  rows : t.regs .int [] "n_rows" = some (fun _ => (p.rows : Int))
  columns : t.regs .int [] "n_cols" = some (fun _ => (p.columns : Int))
  group : t.regs .int [] "group" = some (fun _ => (p.rowsPerProgram : Int))
  one : t.regs .int [] "one" = some (fun _ => (1 : Int))

theorem prelude_run {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) :
    ∃ t, run (withSum M R) (prelude p) s = some t ∧ Ready M p s t (start p s) (initial M p s) := by
  simp [prelude, candidateKernel, candidateSource, backwardCandidateSource, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, numeric, bop, integralFloorDiv,
    numericLt, natLt, withSum, Region.cast, Int.natCast_mul, Int.natCast_add]
  constructor <;> simp [State.setReg, initial, zero, denom, start, stop, withSum, bop, numeric]
  all_goals try rfl
  all_goals funext i
  all_goals try simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]
  all_goals try simp only [← Int.natCast_mul, Int.toNat_natCast]
  by_cases hab : ((s.pids 0 : Int)+1)*p.rowsPerProgram < p.rows
  · rw [if_pos hab, min_eq_left (le_of_lt hab)]
  · rw [if_neg hab, min_eq_right (le_of_not_gt hab)]

theorem body_run {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s t : State α) (row : Nat) (a : Acc α p)
    (h : Ready M p s t row a) :
    ∃ u, run (withSum M R) (body p) (t.setReg "_" .int [] (fun _ => (row : Int))) = some u ∧
      Ready M p s u (row+1) (update M R p row a) := by
  simp [body, candidateKernel, candidateSource, backwardCandidateSource, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, evalOffsets, store, Option.bind, ComputeDType.eraseDType,
    FloatDType.toTileDType, FloatDType.numericDType, ofFloat, toFloat,
    numeric, bop, numericLt, natLt, Region.cast, withSum,
    h.mem, h.sx, h.sdx, h.sdw, h.sdb, h.sdy, h.rows, h.columns, h.group, h.one, h.pid, h.first, h.last, h.cols, h.mask, h.dw, h.db,
    h.xp, h.mp, h.rp, h.dxp, h.dyp, h.denominator]
  constructor <;> simp [State.setReg, h.mem, h.sx, h.sdx, h.sdw, h.sdb, h.sdy, h.rows, h.columns, h.group, h.one, h.pid, h.first, h.last, h.cols, h.mask,
    h.dw, h.db, h.xp, h.mp, h.rp, h.dxp, h.dyp, h.denominator, h.denominator, update, fold_writes_mem,
    initial, zero, stored, dxValue, c1, c2, xhat, xhatTyped, wdy, load, mean, rstd,
    cast, binary, denom, padding, withSum, ofFloat, toFloat, bop, numeric, Nat.add_mul]
  all_goals try norm_num only
  all_goals try rfl
  all_goals try simp only [← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast]
  all_goals try rfl
  all_goals simpa only [vector_bop, wdy_values, padding]


#axiomsClean prelude_run
#axiomsClean body_run
end
end VeriTile.Bench.Repairs.LayerNormLiger.Backward.Candidate
