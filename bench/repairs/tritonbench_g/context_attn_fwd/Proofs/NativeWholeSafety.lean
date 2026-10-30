import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.NativeSourceSafety

namespace VeriTile.Bench.Repairs.ContextAttnFwd.NativeSourceSafety
open VeriTile.Triton FP.Structural NativeFormula FPSuccess FPReference
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

theorem range (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (dots : DotSupport p M) (positive : 0 < p.BN) (initial s : State α)
    (old : Acc p α) (live : Matches p M (FPReference.inputs p r initial) old s)
    (memory : s.mem=initial.mem) (caps : RegionName → Nat)
    (hb : Bounds p r (FPReference.inputs p r initial) caps) (cur stop : Int)
    (reached : ∀ k,k ∈ keySteps p cur stop → k ∈ keySteps p 0
      ((blockMask p (FPReference.inputs p r initial) : Int)*keyEnd p (FPReference.inputs p r initial))) :
    Safety.rangeIntSafe (M.withDefaultPrecision .fp32) caps "start_n" cur stop p.BN
      (parts p r scale).body s := by
  by_cases hc : cur < stop
  · have head : cur ∈ keySteps p cur stop := by
      rw [keySteps,dif_neg (Nat.ne_of_gt positive),dif_pos hc]
      exact List.mem_cons_self
    obtain ⟨mid,ran,live',memory'⟩ := FPReference.body p r scale M dots positive initial s old live memory cur
    rw [Safety.rangeIntSafe,if_neg (Nat.ne_of_gt positive),if_pos hc,ran]
    refine ⟨body p r scale M dots positive initial s old live caps hb cur (reached cur head),?_⟩
    apply range p r scale M dots positive initial mid _ live' memory' caps hb (cur+p.BN) stop
    intro k member
    apply reached
    rw [keySteps,dif_neg (Nat.ne_of_gt positive),dif_pos hc]
    exact List.mem_cons_of_mem _ member
  · rw [Safety.rangeIntSafe,if_neg (Nat.ne_of_gt positive),if_neg hc]
    trivial
termination_by (stop-cur).toNat
decreasing_by omega

theorem loop (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (dots : DotSupport p M) (positive : 0 < p.BN) (initial s : State α)
    (live : Matches p M (FPReference.inputs p r initial) (FPReference.initial p M) s)
    (memory : s.mem=initial.mem) (caps : RegionName → Nat)
    (hb : Bounds p r (FPReference.inputs p r initial) caps) :
    Safety.stepSafe (M.withDefaultPrecision .fp32) caps
      (.forRangeInt "start_n" (parts p r scale).start (parts p r scale).stop
        (parts p r scale).stride (parts p r scale).body) s := by
  have he := live.2.2.1
  have hm := live.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have hstart : evalOp (M.withDefaultPrecision .fp32) none (parts p r scale).start s=
      some (fun _ => (0 : Int)) := rfl
  have hstride : evalOp (M.withDefaultPrecision .fp32) none (parts p r scale).stride s=
      some (fun _ => p.BN) := rfl
  have hstop : evalOp (M.withDefaultPrecision .fp32) none (parts p r scale).stop s=
      some (fun _ => (blockMask p (FPReference.inputs p r initial) : Int)*keyEnd p (FPReference.inputs p r initial)) := by
    change evalOp (M.withDefaultPrecision .fp32) none
      (.mul .int .nil (.castNatToInt (.ref .nat [] "block_mask"))
        (.ref .int [] "block_end_loc")) s = _
    simp only [evalOp_unfold,hm,he,Option.bind,numeric]
    rfl
  rw [Safety.stepSafe,hstart,hstop,hstride]
  refine ⟨?_,?_,?_,?_⟩
  · change Safety.safeOp _ _ _ _ (.constInt 0)
    simp only [Safety.safeOp]
  · change Safety.safeOp _ _ _ _
      (.mul .int .nil (.castNatToInt (.ref .nat [] "block_mask")) (.ref .int [] "block_end_loc"))
    simp only [Safety.safeOp,true_and]
  · change Safety.safeOp _ _ _ _ (.constNat p.BN)
    simp only [Safety.safeOp]
  · exact range p r scale M dots positive initial s _ live memory caps hb 0 _ (fun _ h => h)

theorem source (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (dots : DotSupport p M) (positive : 0 < p.BN) (s : State α)
    (caps : RegionName → Nat) (hb : Bounds p r (FPReference.inputs p r s) caps) :
    TraceSafe (M.withDefaultPrecision .fp32) caps (kernel p r .fp32 scale) s := by
  obtain ⟨pre,ran,live,memory⟩ := FPReference.prelude p r scale M s
  obtain ⟨post,ranLoop,live',memory'⟩ := FPReference.loop p r scale M dots positive s pre live memory
  change Safety.traceSafeList _ _ _ _
  rw [source_split,Safety.traceSafeList_append,Safety.traceSafeList_append]
  refine ⟨⟨prelude p r scale M s caps hb,?_⟩,?_⟩
  · intro mid eq
    have same : mid=pre := Option.some.inj (eq.symm.trans ran)
    subst mid
    rw [Safety.traceSafeList_cons]
    exact ⟨loop p r scale M dots positive s pre live memory caps hb,
      fun _ _ => Safety.traceSafeList_nil _ _ _⟩
  · intro mid eq
    have executed : run (M.withDefaultPrecision .fp32)
        ((parts p r scale).before++[.forRangeInt "start_n" (parts p r scale).start
          (parts p r scale).stop (parts p r scale).stride (parts p r scale).body]) s=some post := by
      rw [run_append,ran]
      simp only [Option.bind_some,run,ranLoop]
      rfl
    have same : mid=post := Option.some.inj (eq.symm.trans executed)
    subst mid
    exact epilogue p r scale M s post _ live' caps hb

#axiomsClean source
end
end VeriTile.Bench.Repairs.ContextAttnFwd.NativeSourceSafety
