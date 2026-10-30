import bench.repairs.tritonbench_g.token_softmax_llama.Proofs.Model
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom.KernelSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Exact two metadata reads and active strided payload/output accesses in
the complete source, evaluated through every actual predecessor state. -/
theorem source_safe {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (positive : 0 < p.width) (s : State α) (bounds : RegionName → Nat)
    (hs : s.pids 0*p.startStride < bounds (Region.cast p.start))
    (hn : s.pids 0*p.lengthStride < bounds (Region.cast p.length))
    (hx : ∀ i : Fin p.width, i.val < count p s →
      s.pids 1*p.inputHeadStride+(startIndex p s+i.val)*p.inputTokenStride < bounds p.input)
    (ho : ∀ i : Fin p.width, i.val < count p s →
      s.pids 1*p.outputHeadStride+(startIndex p s+i.val)*p.outputTokenStride < bounds p.output) :
    TraceSafe M bounds (kernel .fp32 p) s := by
  simp (config := {maxSteps := 1000000}) [TraceSafe, kernel, source, ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold, Safety.stepSafe_unfold, Safety.safeExpr_unfold,
    Safety.safeCompute_unfold, Safety.safeOp_unfold, Safety.safeAddress_unfold,
    Safety.safeMask_unfold, Safety.addressSafe_unfold, Safety.maskActive_unfold,
    step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store,
    State.setReg, numeric, bop, numericLt, natLt, ofFloat, toFloat, Region.cast,
    Option.bind, ComputeDType.eraseDType, TileShape.axisDim, positive]
  exact ⟨hn,hs,hx,ho⟩

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.TokenSoftmaxBloom.KernelSafety
