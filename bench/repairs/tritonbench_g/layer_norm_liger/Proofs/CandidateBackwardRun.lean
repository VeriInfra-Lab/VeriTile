import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardModel
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
  pid : t.regs .nat [] "row_block_id" = some (fun _ => s.pids 0)
  first : t.regs .nat [] "row_start" = some (fun _ => start p s)
  last : t.regs .nat [] "row_end" = some (fun _ => stop p s)
  cols : t.regs .nat [p.block] "cols" = some (fun i => i.1.val)
  mask : t.regs .bool [p.block] "mask" = some (fun i => decide (i.1.val < p.columns))
  dw : t.regs .real [p.block] "dw_row" = some a.dw
  db : t.regs .real [p.block] "db_row" = some a.db
  xp : t.regs .ptr [] "X" = some (fun _ => (p.x,row*p.xStride))
  mp : t.regs .ptr [] "Mean" = some (fun _ => (p.mean,row))
  rp : t.regs .ptr [] "RSTD" = some (fun _ => (p.rstd,row))
  dxp : t.regs .ptr [] "DX" = some (fun _ => (p.dx,row*p.dxStride))
  denominator : t.regs .real [] "denominator" = some (fun _ => denom M p)
  dyp : t.regs .ptr [] "DY" = some (fun _ => (p.dy,row*p.dyStride))

theorem prelude_run {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) :
    ∃ t, run (withSum M R) (prelude p) s = some t ∧ Ready M p s t (start p s) (initial M p s) := by
  simp [prelude, candidateKernel, candidateSource, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, numeric, bop, integralFloorDiv,
    numericLt, natLt, withSum, Region.cast]
  constructor <;> simp [State.setReg, initial, zero, denom, start, stop, withSum, bop, numeric]
  all_goals try rfl
  all_goals funext i
  all_goals try simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]
  by_cases hab : (s.pids 0+1)*p.rowsPerProgram < p.rows
  · rw [if_pos hab, min_eq_left (Nat.le_of_lt hab)]
  · rw [if_neg hab, min_eq_right (Nat.le_of_not_gt hab)]

theorem body_run {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s t : State α) (row : Nat) (a : Acc α p)
    (h : Ready M p s t row a) :
    ∃ u, run (withSum M R) (body p) (t.setReg "_row" .nat [] (fun _ => row)) = some u ∧
      Ready M p s u (row+1) (update M R p row a) := by
  simp [body, candidateKernel, candidateSource, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, evalOffsets, store, Option.bind, ComputeDType.eraseDType,
    FloatDType.toTileDType, FloatDType.numericDType, ofFloat, toFloat,
    numeric, bop, numericLt, natLt, Region.cast, withSum,
    h.mem, h.pid, h.first, h.last, h.cols, h.mask, h.dw, h.db,
    h.xp, h.mp, h.rp, h.dxp, h.dyp, h.denominator]
  constructor <;> simp [State.setReg, h.mem, h.pid, h.first, h.last, h.cols, h.mask,
    h.dw, h.db, h.xp, h.mp, h.rp, h.dxp, h.dyp, h.denominator, h.denominator, update, fold_writes_mem,
    initial, zero, stored, dxValue, c1, c2, xhat, wdy, load, mean, rstd,
    cast, binary, denom, padding, withSum, ofFloat, toFloat, bop, numeric, Nat.add_mul]
  all_goals try norm_num only
  all_goals try rfl
  congr 1
  funext mem i
  rcases i with ⟨i,⟨⟩⟩
  congr! 16 <;> try rfl
  funext j
  rcases j with ⟨j,⟨⟩⟩
  simp [wdy, load, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, padding]

#axiomsClean prelude_run
#axiomsClean body_run
end
end VeriTile.Bench.Repairs.LayerNormLiger.Backward.Candidate
