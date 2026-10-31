import bench.tritonbench_g.swiglu_fwd.Proofs.Kernel
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.SwigluFwd.KernelProof
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluFwd Whole
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000

private theorem trace_last {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionBounds) (stmt : ComputeStmt) (s : State α) :
    Safety.traceSafeList M bounds [stmt] s ↔ Safety.stepSafe M bounds stmt s := by
  rw [Safety.traceSafeList_cons]
  simp

/-- The actual typed source touches only the three declared active row windows.
The source load casts and all register steps remain in the checked trace. -/
theorem trace_safe {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (B row tile : Nat) (s : State α) (bounds : RegionBounds)
    (inputX : ∀ j : Fin B, tile*B+j.val<c.cols → row*c.sx+(tile*B+j.val)<bounds c.X)
    (inputY : ∀ j : Fin B, tile*B+j.val<c.cols → row*c.sy+(tile*B+j.val)<bounds c.Y)
    (output : ∀ j : Fin B, tile*B+j.val<c.cols → row*c.so+(tile*B+j.val)<bounds c.OUT) :
    TraceSafe M bounds (kernel c.X c.Y c.OUT c.sx c.sy c.so c.cols B)
      (tileState M c B row tile s) := by
  simp (maxSteps := 1400000) [trace_last,TraceSafe,kernel,
    ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,store,
    Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,
    tileState,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    Nat.add_assoc]
  exact ⟨inputX,inputY,output⟩

#axiomsClean trace_safe
end
end VeriTile.Bench.Repairs.SwigluFwd.KernelProof
