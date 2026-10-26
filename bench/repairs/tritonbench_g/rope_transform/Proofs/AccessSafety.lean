import bench.repairs.tritonbench_g.rope_transform.Proofs.Grid
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.RopeTransform.SourceSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
noncomputable section
set_option maxHeartbeats 20000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- All reached coefficient loads, input halves and four in-place stores obey
these actual mask-derived windows, independently of floating values. -/
theorem source_safe {α : Type} [Inhabited α] (M : Algebra α)
    (r : Request) (s : State α) (bounds : RegionName → Nat)
    (coefficient : ∀ (sine : Bool) (j : Fin r.width),j.val<r.half →
      s.pids 0%r.sequence*r.half+j.val < bounds (if sine then "S" else "C"))
    (input : ∀ key side (h : Fin (r.paddedHeads key)) (j : Fin r.width),
      h.val<r.heads key → j.val<r.half →
      r.offset key (s.pids 0) h.val (r.column side j.val)<bounds (region key)) :
    TraceSafe M bounds (kernel r) s := by
  simp (maxSteps := 3000000) [TraceSafe,kernel,T.kernel,
    ActualSource.source,Kernels.source,Request.parameters,
    ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    store,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
    numericEq,Region.cast,State.setReg,integralFloorDiv,integralMod,
    FloatDType.numericDType,FloatDType.toTileDType,ofFloat,toFloat,fold_writes_regs,half_eq]
  refine ⟨?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · intro j hj
    exact coefficient Bool.false j hj
  · intro j hj
    exact coefficient Bool.true j hj
  all_goals intro h j hh hj
  all_goals first
    | simpa [Request.offset,Request.column,region,Request.heads,Nat.add_assoc]
        using input Bool.false Bool.false h j hh hj
    | simpa [Request.offset,Request.column,region,Request.heads,Nat.add_assoc]
        using input Bool.true Bool.false h j hh hj
    | simpa [Request.offset,Request.column,region,Request.heads,Nat.add_assoc]
        using input Bool.false Bool.true h j hh hj
    | simpa [Request.offset,Request.column,region,Request.heads,Nat.add_assoc]
        using input Bool.true Bool.true h j hh hj

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.RopeTransform.SourceSafety
