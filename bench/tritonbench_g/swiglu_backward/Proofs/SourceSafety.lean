import bench.tritonbench_g.swiglu_backward.Kernels
import bench.tritonbench_g.swiglu_backward.Proofs.Mathematical_WholeTile
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.SwigluBackward.SourceSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward Whole
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Bounds cover only active lanes of the actual source. Both constexpr
branches, three input casts, and all enabled stores are checked in order. -/
theorem source_safe {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (B row tile : Nat) (R : Bool) (s : State α) (bounds : RegionBounds)
    (hx : ∀ j : Fin B, tile*B+j.val < c.cols → row*c.sx+(tile*B+j.val) < bounds c.X)
    (hy : ∀ j : Fin B, tile*B+j.val < c.cols → row*c.sy+(tile*B+j.val) < bounds c.Y)
    (hd : ∀ j : Fin B, tile*B+j.val < c.cols → row*c.sd+(tile*B+j.val) < bounds c.DOUT)
    (ho : ∀ ch, enabled R ch → ∀ j : Fin B, tile*B+j.val < c.cols →
      row*c.stride ch+(tile*B+j.val) < bounds (c.output ch)) :
    TraceSafe M bounds (c.source B R) (tileState M c B row tile s) := by
  cases R
  all_goals
    simp (maxSteps := 1400000) [TraceSafe, Layout.source, VeriTile.Bench.Repairs.SwigluBackward.kernel,

      ComputeKernel.surfaceBody, Safety.traceSafeList_unfold, Safety.stepSafe_unfold,
      Safety.safeExpr_unfold, Safety.safeCompute_unfold, Safety.safeOp_unfold,
      Safety.safeAddress_unfold, Safety.safeMask_unfold, Safety.addressSafe_unfold,
      Safety.maskActive_unfold, Safety.safeAlg_unfold, run, step,
      FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
      store, Option.bind, ComputeDType.eraseDType, numeric, bop, natLt, tileState,
      ofFloat, toFloat, TileShape.axisDim, Region.cast, State.setReg, Nat.add_assoc]
  · exact ⟨hx,hy,hd,ho .dx (by simp [enabled]),ho .dy (by simp [enabled])⟩
  · exact ⟨hx,hy,hd,ho .dx (by simp [enabled]),ho .dy (by simp [enabled]),
      ho .out (by simp [enabled])⟩

#axiomsClean source_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluBackward.SourceSafety
