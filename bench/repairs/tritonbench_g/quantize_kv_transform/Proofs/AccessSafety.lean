import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.KernelResult
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.AccessSafety
open VeriTile.Triton FP.Structural KernelSource KernelExecution KernelResult
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem source_safe {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (s : State α) (hw : 0 < p.width)
    (dst : Nat) (metadata : (s.mem "destination" (s.pids 0)).read .int = (dst : Int))
    (bounds : RegionName → Nat) (destination : s.pids 0 < bounds "destination")
    (input : ∀ h c, h < p.heads → c < p.columns →
      inputOffset p (s.pids 0) h c < bounds "input")
    (codes : ∀ h c, h < p.heads → c < p.columns → codeOffset p dst h c < bounds "output")
    (scales : ∀ h, h < p.heads → scaleOffset p dst h < bounds "scale") :
    FP.Structural.TraceSafe (backend M convert equal) bounds (kernel p) s := by
  simp [FP.Structural.TraceSafe, kernel, ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold, Safety.stepSafe_unfold,
    Safety.safeExpr_unfold, Safety.safeCompute_unfold, Safety.safeOp_unfold,
    Safety.safeAddress_unfold, Safety.safeMask_unfold, Safety.addressSafe_unfold,
    Safety.maskActive_unfold, step, FP.Structural.evalExpr, evalComputeOp,
    evalOp_unfold, evalOffsets, evalDeltas, store, Option.bind,
    ComputeDType.eraseDType, numeric, numericLt, numericEq, bop, natLt, backend,
    ofFloat, toFloat, Region.cast, State.setReg, TileShape.axisDim, hw, metadata,
    index_toNat, index2_toNat]
  refine ⟨destination, ?_, ?_, ?_⟩
  · intro h c hh hc
    exact input h.val c.val (by omega) (by omega)
  · intro h c hh hc
    exact codes h.val c.val (by omega) (by omega)
  · intro h hh
    exact scales h.val (by omega)

#axiomsClean source_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.QuantizeKvTransform.AccessSafety
