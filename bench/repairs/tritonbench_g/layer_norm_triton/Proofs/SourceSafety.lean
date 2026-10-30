import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceForward
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceGroupedExecution
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceFinalGraph
import VeriTile.Triton.Float.StructuralSafety

/-! Access safety of the three actual typed source stages. Every loop uses the
same indices and intermediate states as Structural.exec; numerical primitive
meanings remain arbitrary. Scalar saved-statistic accesses are unconditional. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceSafety
open VeriTile.Triton FP.Structural SourceModel
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem forward_safe (m : Model α) (stride N B : Nat) (eps : ℝ)
    (positive : 0<N) (covers : N≤B) (s : State α) (bounds : RegionName → Nat)
    (hx : ∀ j,j<N → s.pids 0*stride+j<bounds "X")
    (hw : ∀ j,j<N → j<bounds "W") (hbias : ∀ j,j<N → j<bounds "Bias")
    (hy : ∀ j,j<N → s.pids 0*stride+j<bounds "Y")
    (hm : s.pids 0<bounds "Mean") (hr : s.pids 0<bounds "Rstd") :
    TraceSafe (engine m) bounds
      (TypedSource.forward "X" "Y" "W" "Bias" "Mean" "Rstd" stride N B eps) s := by
  have hB : B≠0 := by omega
  simp [TraceSafe,TypedSource.forward,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    Safety.rangeSafe_unfold,Safety.precisionContext,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,
    range_one _ _ N B positive covers,
    engine,numeric,natLt,numericLt,numericLe,numericEq,bop,
    State.setReg,FP.Structural.store,ComputeDType.eraseDType,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    Region.cast,Option.bind,ofFloat,toFloat,positive,hB,Nat.not_lt.mpr covers,hm,hr]
  exact ⟨fun i => hx i.val,fun i => hw i.val,
    fun i => hbias i.val,fun i => hx i.val,fun i => hy i.val⟩

end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceSafety
