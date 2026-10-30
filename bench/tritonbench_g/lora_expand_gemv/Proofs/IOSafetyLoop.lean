import bench.tritonbench_g.lora_expand_gemv.Proofs.IOSafetyBody

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.IOSafety
open VeriTile.Triton FP.Structural FPValue
open FP.Equational (ReductionPlan)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000

 theorem loop_safe (rest : Algebra Word) (plans : ∀ n,ReductionPlan n) (g : Geometry)
    (limits : RegionName → Nat) (initial current : State Word)
    (h : Branches.Invariant .ieeeValue .singleHalf g.parameters initial current)
    (hl : Branches.length g.parameters=1)
    (hw : ∀ (m : Fin 256) (k : Fin g.block),m.val<Branches.length g.parameters → k.val<g.K →
      Branches.bOffset g.parameters initial m.val k.val<limits "W")
    (hy : ∀ m : Fin 256,m.val<Branches.length g.parameters →
      Branches.outputOffset g.parameters initial m.val<limits "Y") :
    Safety.traceSafeList (Branches.engine rest .ieeeValue plans) limits
      [Branches.loopStmt .singleHalf .single g.parameters] current := by
  rw [Safety.traceSafeList_cons]
  refine ⟨?_,by intro t _; simp⟩
  rw [Branches.loopStmt,Safety.stepSafe_unfold]
  simp only [Safety.safeOp_unfold,evalOp_unfold,h.size]
  change True ∧ True ∧ True ∧ Safety.rangeSafe (Branches.engine rest .ieeeValue plans)
    limits "n" 0 (Branches.length g.parameters) 256 (Branches.body .singleHalf .single g.parameters) current
  simp only [true_and]
  rw [hl,Safety.rangeSafe_unfold]
  norm_num only
  refine ⟨body_safe rest plans g limits initial current h hw hy,?_⟩
  split <;> simp [Safety.rangeSafe_unfold]

end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.IOSafety
