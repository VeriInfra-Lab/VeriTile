import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IOSafety

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.IOSafety
open VeriTile.Triton FP.Structural FPValue
open FP.Equational (ReductionPlan)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false

 theorem constIf (M : Algebra Word) (limits : RegionName → Nat) (b : Bool)
    (yes rest : List ComputeStmt) (s : State Word) :
    Safety.traceSafeList M limits (.ifThen (.constBool b) yes::rest) s ↔
    Safety.traceSafeList M limits ((if b then yes else [])++rest) s := by
  cases b <;>
    simp [Safety.traceSafeList_cons,Safety.traceSafeList_append,Safety.stepSafe_unfold,
      Safety.safeOp_unfold,step,evalOp_unfold,Option.bind]

 theorem constIfElse (M : Algebra Word) (limits : RegionName → Nat) (b : Bool)
    (yes no rest : List ComputeStmt) (s : State Word) :
    Safety.traceSafeList M limits (.ifThenElse (.constBool b) yes no::rest) s ↔
    Safety.traceSafeList M limits ((if b then yes else no)++rest) s := by
  cases b <;>
    simp [Safety.traceSafeList_cons,Safety.traceSafeList_append,Safety.stepSafe_unfold,
      Safety.safeOp_unfold,step,evalOp_unfold,Option.bind]

/-- Constant dtype control is removed only after proving preservation of both
actual execution states and their access-safety obligations. -/
theorem active_safe_iff (rest : Algebra Word) (plans : ∀ n, ReductionPlan n)
    (g : Geometry) (limits : RegionName → Nat) (s : State Word) :
    Safety.traceSafeList (Branches.engine rest .ieeeValue plans) limits
      (Branches.active .singleHalf .single g.parameters) s ↔
    Safety.traceSafeList (Branches.engine rest .ieeeValue plans) limits
      (Branches.setup .singleHalf .single g.parameters ++
        [Branches.loopStmt .singleHalf .single g.parameters]) s := by
  conv_lhs => rw [← List.take_append_drop 4 (Branches.active .singleHalf .single g.parameters)]
  rw [Branches.setup,List.append_assoc]
  simp only [Safety.traceSafeList_append]
  apply and_congr_right
  intro _
  apply forall_congr'
  intro t
  apply imp_congr_right
  intro _
  simp [Branches.active,Branches.castBranch,Branches.loopStmt,Branches.body,
    Branches.source,VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.source,
    ComputeKernel.surfaceBody,Geometry.parameters,constIfElse,constIf]
  rw [← Safety.traceSafeList_append]
  rfl

end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.IOSafety
