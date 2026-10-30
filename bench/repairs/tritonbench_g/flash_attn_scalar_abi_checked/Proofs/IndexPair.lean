import bench.repairs.tritonbench_g.flash_attn_scalar_abi_checked.Kernels
import VeriTile.Triton.Float.StructuralSafety
import VeriTile.Triton.Float.ProfiledRewrite

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.IndexPair
open VeriTile Triton FP.Structural FlashAttn FlashAttn.ActualSource
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 3000000
variable {α : Type} [Inhabited α]

def prelude (swapped : Bool) (p : Parameters) : List ComputeStmt :=
  (if swapped then ActualSource.indexCommuted p else ActualSource.kernel p).surfaceBody.take 8

def suffix (p : Parameters) : List ComputeStmt := (ActualSource.kernel p).surfaceBody.drop 8

theorem source_split (swapped : Bool) (p : Parameters) :
    (if swapped then ActualSource.indexCommuted p else ActualSource.kernel p).surfaceBody =
      prelude swapped p ++ suffix p := by
  cases swapped <;> rfl

theorem original_split (p : Parameters) :
    (ActualSource.kernel p).surfaceBody = prelude Bool.false p ++ suffix p :=
  source_split Bool.false p

theorem candidate_split (p : Parameters) :
    (ActualSource.indexCommuted p).surfaceBody = prelude Bool.true p ++ suffix p :=
  source_split Bool.true p

theorem prelude_runs (M : Algebra α) (p : Parameters) (s : State α) :
    run M (prelude Bool.true p) s = run M (prelude Bool.false p) s := by
  have comm (a b : TileIndex [] → Nat) :
      FP.Structural.bop (fun x y : Nat => x * y) Broadcast.nil a b =
        FP.Structural.bop (fun x y : Nat => x * y) Broadcast.nil b a := by
    funext i
    simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Nat.mul_comm]
  simp [prelude,ActualSource.kernel,ActualSource.indexCommuted,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalOp_unfold,numeric,integralFloorDiv,integralMod,comm]

theorem prelude_safe (M : Algebra α) (bounds : RegionName → Nat)
    (p : Parameters) (swapped : Bool) (s : State α) :
    Safety.traceSafeList M bounds (prelude swapped p) s := by
  cases swapped <;> simp [prelude,ActualSource.kernel,ActualSource.indexCommuted,
    ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr,Safety.safeOp_unfold,step,evalExpr,evalOp_unfold,numeric,
    integralFloorDiv,integralMod,FP.Structural.bop,State.setReg]

theorem runs (M : Algebra α) (p : Parameters) (s : State α) :
    FP.Structural.exec M (ActualSource.indexCommuted p) s = FP.Structural.exec M (ActualSource.kernel p) s := by
  change run M (ActualSource.indexCommuted p).surfaceBody s =
    run M (ActualSource.kernel p).surfaceBody s
  rw [candidate_split,original_split,run_append,run_append,prelude_runs]

theorem safe (M : Algebra α) (bounds : RegionName → Nat) (p : Parameters) (s : State α) :
    TraceSafe M bounds (ActualSource.indexCommuted p) s ↔
      TraceSafe M bounds (ActualSource.kernel p) s := by
  change Safety.traceSafeList M bounds _ s ↔ Safety.traceSafeList M bounds _ s
  rw [candidate_split,original_split,Safety.traceSafeList_append,Safety.traceSafeList_append,
    prelude_runs]
  simp only [prelude_safe,true_and]

#axiomsClean runs
#axiomsClean safe
end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.IndexPair
