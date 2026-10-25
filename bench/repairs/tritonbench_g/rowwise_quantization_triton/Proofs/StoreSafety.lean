import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.StoreSchedule

namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.StoreSchedule
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- The early maximum store has exactly the original backed output address;
all following computations use the already loaded register tile. -/
theorem candidate_safe (M : Algebra α) (round : α → BitVec 64) (equal : α → α → Bool)
    (columns width : Nat) (s : State α) (hw : 0 < width) (bounds : RegionName → Nat)
    (hx : ∀ j : Fin width, j.val < columns → s.pids 0*columns+j.val < bounds "x")
    (hq : ∀ j : Fin width, j.val < columns → s.pids 0*columns+j.val < bounds "q")
    (hm : s.pids 0 < bounds "m") :
    QuantizeIntegerStage.TraceSafe (KernelExecution.backend M equal) (some round) bounds
      ⟨Candidate.source "x" "q" "m" columns width,fun _ => none,QuantizeIntegerStage.outputRegister⟩ s := by
  simp (maxSteps := 1500000) [QuantizeIntegerStage.TraceSafe,
    Candidate.source,earlyMaximumSource,ComputeKernel.surfaceBody,
    QuantizeIntegerStage.Safety.traceSafeList_unfold,QuantizeIntegerStage.Safety.stepSafe_unfold,
    QuantizeIntegerStage.Safety.safeExpr_unfold,QuantizeIntegerStage.step,QuantizeIntegerStage.evalExpr,
    QuantizeIntegerStage.store,QuantizeIntegerStage.storedInteger,QuantizeIntegerStage.outputRegister,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,FP.Structural.store,numeric,numericLt,
    numericEq,natLt,bop,KernelExecution.backend,State.setReg,ComputeDType.eraseDType,
    ofFloat,toFloat,Region.cast,Option.bind,TileShape.axisDim,hw,hm]
  simp only [← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast]
  exact ⟨hx,hq⟩

#axiomsClean candidate_safe
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.StoreSchedule
