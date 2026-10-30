import bench.repairs.tritonbench_g.parallel_attention.Proofs.IOSafetyForward

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.IOSafety
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem pure_list (M : Algebra α) (bounds : RegionName → Nat)
    (code : List ComputeStmt) (safe : ∀ st∈code,∀ s,Safety.stepSafe M bounds st s)
    (s : State α) : Safety.traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => exact Safety.traceSafeList_nil _ _ _
  | cons st rest ih =>
    apply Safety.traceSafeList_cons_intro (safe st (by simp) s)
    intro t _
    exact ih (fun st h=>safe st (by simp [h])) t

private theorem second_body_tail (M : Algebra α) (D : DotPrimitive α) (c : ForwardConfig)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (withDot M D) bounds ((secondBody c.BTL c.BTS c.BK c.BV).drop 2) s := by
  apply pure_list
  intro st hs t
  simp [secondBody,forwardSource,ComputeKernel.surfaceBody] at hs
  rcases hs with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
  all_goals simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
  all_goals try simp only [safeBlockPtr2]
  all_goals simp [Safety.safeOp_unfold]

theorem forward_second_body (M : Algebra α) (D : DotPrimitive α) (c : ForwardConfig)
    (bounds : RegionName → Nat) (s t : State α) (start : Nat)
    (acc : ForwardAccumulator α c.BTL c.BV) (h : DiagonalReady M c s t start acc)
    (hb : ForwardBounds c s bounds) :
    Safety.traceSafeList (withDot M D) bounds (secondBody c.BTL c.BTS c.BK c.BV) t := by
  have hf:=forward_first_body M D c bounds s t start acc h.toForwardReady hb
  rw [←List.take_append_drop 2 (firstBody c.BTL c.BTS c.BK c.BV),Safety.traceSafeList_append] at hf
  rw [←List.take_append_drop 2 (secondBody c.BTL c.BTS c.BK c.BV),Safety.traceSafeList_append]
  exact ⟨hf.1,fun u _=>second_body_tail M D c bounds u⟩

theorem forward_between (M : Algebra α) (D : DotPrimitive α) (c : ForwardConfig)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (withDot M D) bounds (betweenLoops c) s := by
  apply pure_list
  intro st hs t
  simp [betweenLoops,forwardKernel,forwardSource,ComputeKernel.surfaceBody] at hs
  rcases hs with rfl|rfl|rfl|rfl|rfl
  all_goals simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
  all_goals try simp only [safeBlockPtr2]
  all_goals simp [Safety.safeOp_unfold]

theorem forward_first_range (M : Algebra α) (D : DotPrimitive α) (c : ForwardConfig)
    (bounds : RegionName → Nat) (s t : State α) (start stop : Nat)
    (acc : ForwardAccumulator α c.BTL c.BV) (h : ForwardReady M c s t start acc)
    (hb : ForwardBounds c s bounds) :
    Safety.rangeSafe (withDot M D) bounds "_i" start stop c.BTS
      (firstBody c.BTL c.BTS c.BK c.BV) t := by
  apply Safety.rangeSafe_of_invariant (withDot M D) bounds "_i" stop c.BTS _
    (fun q u=>∃ a,ForwardReady M c s u q a)
  · rintro q _ u ⟨a,hu⟩
    exact forward_first_body M D c bounds s _ q a (forwardReady_set_index M c s u q a hu q) hb
  · rintro q _ u ⟨a,hu⟩ v ran
    obtain ⟨w,hw,hready⟩:=first_body_run M D c s _ q a (forwardReady_set_index M c s u q a hu q)
    rw [hw] at ran
    cases Option.some.inj ran
    exact ⟨_,hready⟩
  · exact ⟨acc,h⟩

theorem forward_second_range (M : Algebra α) (D : DotPrimitive α) (c : ForwardConfig)
    (bounds : RegionName → Nat) (s t : State α) (start stop : Nat)
    (acc : ForwardAccumulator α c.BTL c.BV) (h : DiagonalReady M c s t start acc)
    (hb : ForwardBounds c s bounds) :
    Safety.rangeSafe (withDot M D) bounds "_i" start stop c.BTS
      (secondBody c.BTL c.BTS c.BK c.BV) t := by
  apply Safety.rangeSafe_of_invariant (withDot M D) bounds "_i" stop c.BTS _
    (fun q u=>∃ a,DiagonalReady M c s u q a)
  · rintro q _ u ⟨a,hu⟩
    exact forward_second_body M D c bounds s _ q a (diagonalReady_set_index M c s u q a hu q) hb
  · rintro q _ u ⟨a,hu⟩ v ran
    obtain ⟨w,hw,hready⟩:=second_body_run M D c s _ q a (diagonalReady_set_index M c s u q a hu q)
    rw [hw] at ran
    cases Option.some.inj ran
    exact ⟨_,hready⟩
  · exact ⟨acc,h⟩

#axiomsClean forward_second_body
#axiomsClean forward_between
#axiomsClean forward_first_range
#axiomsClean forward_second_range
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.IOSafety
