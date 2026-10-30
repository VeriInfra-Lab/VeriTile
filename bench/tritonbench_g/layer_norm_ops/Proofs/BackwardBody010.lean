import bench.tritonbench_g.layer_norm_ops.Proofs.BackwardInvariant
import bench.tritonbench_g.layer_norm_ops.Proofs.MemoryHelpers
namespace VeriTile.Bench.Repairs.LayerNormOps.Backward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits (fold_writes_mem fold_writes_regs)
open Forward (fold_writes_undef vector_bop region_name_eq)
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option pp.maxSteps 1200
set_option linter.unusedSimpArgs false

theorem body010 {α : Type} [Inhabited α] (M : Algebra α)
    (p : BackwardParameters) (s t : State α) (row : Nat) (a : Acc α p)
    (h : Ready p s t row a)
    (hr : p.rms = Bool.false) (hd : p.hasDres = Bool.true) (hs : p.storeDres = Bool.false)
    :
    ∃ u, run M (body p) (t.setReg "row" .nat [] (fun _ => row)) = some u ∧
      Ready p s u (row+1) (update M p row a) := by
  cases hb : p.hasBias <;> cases hy : p.recompute
  all_goals
    simp [body, backwardKernel, backwardSource, ComputeKernel.surfaceBody, run, step, evalExpr,
      evalComputeOp, evalOp_unfold, evalOffsets, store, Option.bind, ComputeDType.eraseDType,
      FloatDType.toTileDType, FloatDType.numericDType, ofFloat, toFloat,
      numeric, bop, numericLt, natLt, Region.cast, hr, hd, hs, hb, hy,
      h.mem, h.pid, h.first, h.last, h.cols, h.mask, h.dw, h.db, h.weight, h.bias,
      h.xp, h.yp, h.dxp, h.dyp, h.dres, h.dresIn]
    constructor <;> simp [h.mem, h.pid, h.first, h.last, h.cols, h.mask,
      h.dw, h.db, h.weight, h.bias, h.xp, h.yp, h.dxp, h.dyp, h.dres, h.dresIn,
      update, recomputed, scatter, fold_writes_mem, initial,
      Forward.stored, dxValue, dxBase, yValue, c1, c2, xhat, wdy, dy, loaded, meanValue, rstdValue,
      Forward.cast, Forward.binary, Forward.unhintedBinary, Forward.literal, Forward.padding, average,
      ofFloat, toFloat, bop, numeric, Nat.add_mul, hr, hd, hs, hb, hy]
    all_goals try norm_num only
    all_goals try rfl
    all_goals try simp [bop, Broadcast.leftIndex, Broadcast.rightIndex]

    all_goals try delta wdy dy loaded
    all_goals try simp [Forward.cast, Forward.binary, Forward.literal, Forward.padding]
    all_goals try rfl

#axiomsClean body010
end
end VeriTile.Bench.Repairs.LayerNormOps.Backward
