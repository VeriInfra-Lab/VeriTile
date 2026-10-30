import bench.repairs.tritonbench_g.decay_cumsum.Proofs.FPTile
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.DecayCumsum.IOSafety
open VeriTile.Triton FP.Structural FP.ProfiledRewrite FPPrefix FPOps Geometry
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Every actual prologue access is the prepare kernel's last-gate read. -/
theorem prefix_safe (M : Algebra α) (kind : Kind) (h : Call) (p : Phase)
    (positive : 0<p.span) (prog : Program h p) (bounds : RegionName → Nat) (u : State α)
    (hf : u.pids 0=prog.feature) (hc : u.pids 1=prog.chunk) (hr : u.pids 2=prog.row)
    (hg : ∀ j : Fin h.BK, off h p prog (p.span-1) j<bounds "g") :
    Safety.traceSafeList (engine M) bounds (prefixSource kind h p) u := by
  have last : prog.chunk*p.span+p.span-1=prog.chunk*p.span+(p.span-1) := by omega
  cases kind <;>
    simp (config := { maxSteps := 500000 }) [prefixSource,ids,pcode,forwardPointer,backwardPointer,
      zeroCode,maskCode,headOp,featureOp,chunkOp,lastOp,lastOffset,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      step,run,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      hf,hc,hr,State.setReg,engine,numeric,natLt,bop,integralFloorDiv,ComputeDType.eraseDType,
      Broadcast.leftIndex,Broadcast.rightIndex,Region.cast]
  intro j
  convert hg j using 1 <;> simp only [off,last,Nat.add_mul]
  all_goals ring

#axiomsClean prefix_safe
end
end VeriTile.Bench.Repairs.DecayCumsum.IOSafety
