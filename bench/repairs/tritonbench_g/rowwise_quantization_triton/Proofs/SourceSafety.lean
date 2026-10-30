import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.IntegerBinding

namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.SourceSafety
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem fold_regs {ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active] (r : RegionName)
    (off : ι → Nat) (v : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => if active i then t.write r (off i) (v i) else t) s).regs=s.regs := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih => simp only [List.foldl_cons,ih];split <;> rfl

/-- Every actual masked load/INT8-observed store and unmasked FP16 maximum
store is backed. The llrint child is checked by the shared integer interpreter. -/
theorem source_safe (M : Algebra α) (round : α → BitVec 64) (equal : α → α → Bool)
    (p : KernelExecution.Parameters) (s : State α) (hw : 0<p.width)
    (bounds : RegionName → Nat)
    (hx : ∀ j : Fin p.width, j.val<p.columns → s.pids 0*p.columns+j.val<bounds p.x)
    (hq : ∀ j : Fin p.width, j.val<p.columns → s.pids 0*p.columns+j.val<bounds p.out)
    (hm : s.pids 0<bounds p.maxOut) :
    QuantizeIntegerStage.TraceSafe (KernelExecution.backend M equal) (some round)
      bounds (IntegerBinding.typedKernel p) s := by
  simp (maxSteps := 1500000) [QuantizeIntegerStage.TraceSafe,IntegerBinding.typedKernel,
    KernelExecution.kernel,source,ComputeKernel.surfaceBody,
    QuantizeIntegerStage.Safety.traceSafeList_unfold,QuantizeIntegerStage.Safety.stepSafe_unfold,
    QuantizeIntegerStage.Safety.safeExpr_unfold,QuantizeIntegerStage.step,QuantizeIntegerStage.evalExpr,
    QuantizeIntegerStage.store,QuantizeIntegerStage.storedInteger,QuantizeIntegerStage.outputRegister,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,FP.Structural.store,numeric,numericLt,
    numericEq,natLt,bop,KernelExecution.backend,State.setReg,ComputeDType.eraseDType,
    ofFloat,toFloat,Region.cast,Option.bind,TileShape.axisDim,hw,fold_regs,hm]
  simp only [← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast]
  exact ⟨hx,hq⟩

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.SourceSafety
