import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.SourceSafety

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.SourceSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite Safety FusedRwkv6Kernel Source
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 2000000
variable {α : Type} [Inhabited α]

theorem loop_safe (M : Algebra α) (c : Shape) (initial current : State α) (scale : α)
    (n : Nat) (hn : n≤c.T) (ctx : Context M c initial scale n current)
    (bounds : RegionName → Nat) (hb : Bounds c initial bounds) :
    rangeSafe (engine M) bounds "_i" n c.T 1 (parts c).body current := by
  rw [rangeSafe_unfold]
  by_cases live : n<c.T
  · simp only [show (1 : Nat)≠0 by decide,if_false,if_pos live]
    rw [(bodyExec M c initial current scale n ctx).property]
    refine ⟨body_safe M c initial current scale n live ctx bounds hb,?_⟩
    exact loop_safe M c initial (bodyExec M c initial current scale n ctx).val scale (n+1)
      (by omega) (body_context M c initial current scale n ctx live) bounds hb
  · simp only [show (1 : Nat)≠0 by decide,if_false,if_neg live]
termination_by c.T-n
decreasing_by omega

theorem source_safe (M : Algebra α) (c : Shape) (s : State α) (scale : α)
    (scalar : s.regs .real [] "scale"=some (fun _ => scale))
    (bounds : RegionName → Nat) (hb : Bounds c s bounds) :
    TraceSafe (engine M) bounds (kernel c) s := by
  let start := entry M c s
  obtain ⟨middle,run,ctx⟩ := loopExec M c s start.val scale 0 (by omega) (entry_context M c s scale scalar)
  unfold TraceSafe
  rw [source_body,List.append_assoc,traceSafeList_append]
  refine ⟨entry_safe M c s bounds hb,?_⟩
  intro next hnext
  have heq : next=start.val := Option.some.inj (hnext.symm.trans start.property)
  subst next
  have ran : FP.Structural.step (engine M) (.forRange "_i" 0 c.T 1 (parts c).body) start.val=some middle := by
    simpa only [FP.Structural.step] using run
  apply traceSafeList_cons_of_step (t:=middle) (executed:=ran)
  · rw [stepSafe_unfold]
    exact loop_safe M c s start.val scale 0 (by omega) (entry_context M c s scale scalar) bounds hb
  · exact finish_safe M c s middle scale (ctx.final M c s middle scale) bounds hb

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.SourceSafety
