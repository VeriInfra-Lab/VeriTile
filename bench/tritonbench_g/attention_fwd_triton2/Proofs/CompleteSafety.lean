import bench.tritonbench_g.attention_fwd_triton2.Proofs.SourceSafety

namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.SourceSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole001 Whole001.Source
noncomputable section
variable {α : Type} [Inhabited α]

theorem range_safe (M : Algebra α) (dot : Dot α) (p : Parameters) (bounds : RegionName → Nat)
    (initial current : State α) (row : Nat) (a : Acc α)
    (ctx : Context M p initial row a current) (hb : Bounds p initial bounds) :
    Safety.rangeSafe (engine (withDot M dot)) bounds "start_n" row p.length 64 (parts p).body current := by
  rw [Safety.rangeSafe_unfold,if_neg (by decide : (64 : Nat)≠0)]
  by_cases inside : row<p.length
  · rw [if_pos inside]
    refine ⟨body_safe M dot p bounds initial current row a ctx inside hb,?_⟩
    rw [(bodyExec M dot p initial current row a ctx).property]
    exact range_safe M dot p bounds initial (bodyExec M dot p initial current row a ctx).val
      (row+64) (update M dot p initial row a) (body_context M dot p initial current row a ctx) hb
  · rw [if_neg inside]
    trivial
termination_by p.length-row

theorem source_safe (M : Algebra α) (dot : Dot α) (p : Parameters) (bounds : RegionName → Nat)
    (initial : State α) (hb : Bounds p initial bounds) :
    TraceSafe (engine (withDot M dot)) bounds (kernel p) initial := by
  rw [TraceSafe,source_body,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨entry_safe M dot p bounds initial hb,?_⟩
  intro middle ran
  have same : middle=(entry M dot p initial).val :=
    Option.some.inj (ran.symm.trans (entry M dot p initial).property)
  subst middle
  obtain ⟨final,executed,row,ctx⟩ := loopExec M dot p initial (entry M dot p initial).val
    (entry_context M dot p initial)
  apply Safety.traceSafeList_cons_of_step (t:=final)
  · rw [Safety.stepSafe_unfold]
    change Safety.rangeSafe (engine (withDot M dot)) bounds "start_n" 0 p.length 64 (parts p).body _
    exact range_safe M dot p bounds initial _ 0 _ (entry_context M dot p initial) hb
  · rw [step]
    change range (engine (withDot M dot)) "start_n" 0 p.length 64 (parts p).body _=some final
    exact executed
  · exact post_safe M dot p bounds initial final row _ ctx hb

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.SourceSafety
