import bench.tritonbench_g.flash_attn.Proofs.IOSafetyFinish
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.IOSourceSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite FlashAttn.ActualSource
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem range_safe (M : Algebra α) (dot : Dot α) (p : Parameters) (scale : α)
    (initial : State α) (bounds : RegionName → Nat) (hb : Bounds p initial bounds)
    (row : Nat) (current : State α) (a : Acc p α)
    (ctx : Context M p initial scale row a current) :
    Safety.rangeSafe (engine (withDot M dot)) bounds "start_n" row (stop p initial) 64 (parts p).body current := by
  rw [Safety.rangeSafe_unfold]
  simp only [show (64 : Nat)≠0 by decide,↓reduceIte]
  split
  · refine ⟨body_safe M dot p scale initial current row a ctx bounds hb,?_⟩
    rw [(bodyExec M dot p scale initial current row a ctx).property]
    exact range_safe M dot p scale initial bounds hb (row+64) _ _
      (body_context M dot p scale initial current row a ctx)
  · trivial
termination_by stop p initial-row
decreasing_by omega

theorem source_safe (M : Algebra α) (dot : Dot α) (p : Parameters) (scale : α)
    (s : State α) (bounds : RegionName → Nat) (hb : Bounds p s bounds) :
    TraceSafe (engine (withDot M dot)) bounds (kernel p) (bindScale scale s) := by
  change Safety.traceSafeList (engine (withDot M dot)) bounds _ _
  rw [source_parts,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨entry_safe M dot p scale s bounds hb,?_⟩
  intro start he
  have eq := Option.some.inj ((entry M dot p scale s).property.symm.trans he)
  subst start
  obtain ⟨mid,hm,row,hc⟩ := loopExec M dot p scale s _ (entry_context M dot p scale s)
  apply Safety.traceSafeList_cons_of_step (t := mid)
  · simp only [Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,
      (entry_bounds M dot p scale s).1,(entry_bounds M dot p scale s).2,Option.bind,Option.bind_some,true_and]
    exact range_safe M dot p scale s bounds hb 0 _ _ (entry_context M dot p scale s)
  · simpa [step,evalOp_unfold,(entry_bounds M dot p scale s).1,(entry_bounds M dot p scale s).2] using hm
  · exact finish_safe M dot p scale s mid row _ hc bounds hb

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.IOSourceSafety
