import bench.tritonbench_g.swiglu_triton.Proofs.DirectTile
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.SwigluTriton.SourceSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluTriton
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Only active source loads and enabled output stores require capacity.
This checks the actual forward/backward source statements in execution order. -/
theorem source_safe {α : Type} [Inhabited α] (M : Algebra α) (backward : Bool)
    (block cols row : Nat) (s : State α) (bounds : RegionName → Nat)
    (ha : ∀ j : Fin block, j.val < cols → row*cols+j.val < bounds "A")
    (hb : ∀ j : Fin block, j.val < cols → row*cols+j.val < bounds "B")
    (hdc : backward = Bool.true → ∀ j : Fin block, j.val < cols → row*cols+j.val < bounds "DC")
    (hc : backward = Bool.false → ∀ j : Fin block, j.val < cols → row*cols+j.val < bounds "C") :
    TraceSafe M bounds (directSource backward block cols) (DirectTile.rowState row s) := by
  cases backward
  all_goals
    simp (maxSteps := 900000) [TraceSafe, Safety.traceSafeList_unfold,
      Safety.stepSafe_unfold, Safety.safeExpr_unfold, Safety.safeCompute_unfold,
      Safety.safeOp_unfold, Safety.safeAddress_unfold, Safety.safeMask_unfold,
      Safety.addressSafe_unfold, Safety.maskActive_unfold, Safety.safeAlg_unfold,
      directSource, forwardKernel, backwardKernel, typedFwd,VeriTile.Bench.Repairs.SwigluTriton.forwardKernel,VeriTile.Bench.Repairs.SwigluTriton.forwardIndexCommuted,VeriTile.Bench.Repairs.SwigluTriton.backwardKernel,VeriTile.Bench.Repairs.SwigluTriton.backwardIndexCommuted, typedBwd, typedBwd_fp32_fp32,VeriTile.Bench.Repairs.SwigluTriton.forwardKernel,VeriTile.Bench.Repairs.SwigluTriton.forwardIndexCommuted,VeriTile.Bench.Repairs.SwigluTriton.backwardKernel,VeriTile.Bench.Repairs.SwigluTriton.backwardIndexCommuted,
      ComputeKernel.surfaceBody, run, step, FP.Structural.evalExpr, evalComputeOp,
      evalOp_unfold, evalOffsets, evalDeltas, store, Option.bind, ComputeDType.eraseDType,
      numeric, bop, natLt, DirectTile.rowState, ofFloat, toFloat, TileShape.axisDim,
      Region.cast, State.setReg, ha, hb, hdc, hc]
  · exact ⟨ha, hb, hc rfl⟩
  · exact ⟨hdc rfl, ha, hb, ha, hb⟩

#axiomsClean source_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluTriton.SourceSafety
