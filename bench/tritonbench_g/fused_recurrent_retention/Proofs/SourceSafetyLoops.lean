import bench.tritonbench_g.fused_recurrent_retention.Proofs.SourceSafetyForward
import bench.tritonbench_g.fused_recurrent_retention.Proofs.SourceSafetyBackward
import bench.tritonbench_g.fused_recurrent_retention.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.SourceSafety
open VeriTile.Triton FP.Structural Quoted
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 18000
variable {α : Type} [Inhabited α]

theorem forward_range_safe (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (seed final : Bool) (initial entry : State (Staged.Node α)) (pid : ProgramId)
    (hBK : 0<s.BK) (covered : plans.output.Covers s.BK)
    (h : ForwardInvariant a s b initial seed pid 0 entry)
    (bounds : RegionName → Nat) (hb : ForwardBounds s b pid seed final bounds) :
    Safety.rangeSafe (Staged.algebra a plans) bounds "_t" 0 s.T 1
      (Program.forwardParts (layout s b) forwardRegions seed final).body entry := by
  apply Safety.rangeSafe_of_invariant (Staged.algebra a plans) bounds "_t" s.T 1 _
    (fun t u => ForwardInvariant a s b initial seed pid t u)
  · intro t ht u inv
    exact forward_body_safe a s b plans seed final initial u pid t inv ht bounds hb
  · intro t _ u inv v ran
    rw [forward_body_values a s b plans seed final initial u pid t hBK covered inv] at ran
    cases Option.some.inj ran
    exact forwardBodyResult_invariant a s b plans initial u seed pid t inv
  · exact h

theorem forward_safe (a : Arithmetic α) (s : Shape) (b : InputBuffers) (p : PlanBinding s)
    (seed final : Bool) (initial : State (Staged.Node α)) (shape : s.Admitted)
    (hscale : initial.regs .real [] "scale"=some (fun _ => Staged.tagged .scale (a.scale s.DK)))
    (bounds : RegionName → Nat) (hb : ForwardBounds s b (launch initial) seed final bounds) :
    TraceSafe (Staged.algebra a p.plans) bounds (Program.forward (layout s b) forwardRegions seed final) initial := by
  obtain ⟨entry,he,inv,_⟩ := forward_prefix_values a s b p.plans seed final initial hscale
  obtain ⟨ran,lastInv⟩ := forward_range_values a s b p.plans initial seed final (launch initial) 0 entry
    (by omega) (block_positive s shape).1 p.output.covers inv
  change Safety.traceSafeList _ _ _ _
  rw [Program.forward_split,List.append_assoc]
  simp only [List.singleton_append]
  rw [Safety.traceSafeList_append]
  refine ⟨forward_prefix_safe a s b p.plans seed final initial hscale bounds hb,?_⟩
  intro u hu
  rw [he] at hu
  cases Option.some.inj hu
  apply Safety.traceSafeList_cons_of_step
  · rw [Safety.stepSafe_unfold]
    exact forward_range_safe a s b p.plans seed final initial entry (launch initial)
      (block_positive s shape).1 p.output.covers inv bounds hb
  · simpa only [step] using ran
  · exact forward_final_safe a s b p.plans seed final initial _ (launch initial) lastInv bounds hb

theorem first_range_safe (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (seed final : Bool) (initial entry : State (Staged.Node α)) (pid : ProgramId)
    (hBV : 0<s.BV) (covered : plans.dq.Covers s.BV)
    (h : BackwardFirstInvariant a s b initial seed pid 0 entry)
    (bounds : RegionName → Nat) (hb : BackwardBounds s b pid seed final bounds) :
    Safety.rangeSafe (Staged.algebra a plans) bounds "i" 0 s.T 1
      (Program.backwardFirst (layout s b) backwardRegions seed final).body entry := by
  apply Safety.rangeSafe_of_invariant (Staged.algebra a plans) bounds "i" s.T 1 _
    (fun t u => BackwardFirstInvariant a s b initial seed pid t u)
  · intro t ht u inv
    exact backward_first_safe a s b plans seed final initial u pid t inv ht bounds hb
  · intro t _ u inv v ran
    rw [backward_first_values a s b plans seed final initial u pid t hBV covered inv] at ran
    cases Option.some.inj ran
    exact backwardFirstBodyResult_invariant a s b plans initial u seed pid t inv
  · exact h

theorem second_range_safe (a : Arithmetic α) (s : Shape) (b : InputBuffers) (plans : Plans)
    (seed final : Bool) (initial entry : State (Staged.Node α)) (pid : ProgramId)
    (hBK : 0<s.BK) (hBV : 0<s.BV) (coveredK : plans.dk.Covers s.BV) (coveredV : plans.dv.Covers s.BK)
    (h : BackwardSecondInvariant a s b initial final pid 0 entry)
    (bounds : RegionName → Nat) (hb : BackwardBounds s b pid seed final bounds) :
    Safety.rangeSafe (Staged.algebra a plans) bounds "_t" 0 s.T 1
      (Program.backwardSecond (layout s b) backwardRegions seed final).body entry := by
  apply Safety.rangeSafe_of_invariant (Staged.algebra a plans) bounds "_t" s.T 1 _
    (fun t u => BackwardSecondInvariant a s b initial final pid t u)
  · intro t ht u inv
    exact backward_second_safe a s b plans seed final initial u pid t inv ht bounds hb
  · intro t ht u inv v ran
    rw [reverse_body_values a s b plans seed final initial u pid t (by omega) hBK hBV coveredK coveredV inv] at ran
    cases Option.some.inj ran
    exact reverseBodyResult_invariant a s b plans initial u final pid t inv
  · exact h

theorem backward_safe (a : Arithmetic α) (s : Shape) (b : InputBuffers) (p : PlanBinding s)
    (seed final : Bool) (initial : State (Staged.Node α)) (shape : s.Admitted)
    (hscale : initial.regs .real [] "scale"=some (fun _ => Staged.tagged .scale (a.scale s.DK)))
    (bounds : RegionName → Nat) (hb : BackwardBounds s b (launch initial) seed final bounds) :
    TraceSafe (Staged.algebra a p.plans) bounds (Program.backward (layout s b) backwardRegions seed final) initial := by
  obtain ⟨entry,he,inv,_⟩ := backward_prefix_values a s b p.plans seed final initial hscale
  obtain ⟨ran,lastInv⟩ := backward_first_range_values a s b p.plans initial seed final (launch initial) 0 entry
    (by omega) (block_positive s shape).2 p.dq.covers inv
  obtain ⟨middle,hm,secondInv,_⟩ := backward_middle_values a s b p.plans seed final initial _ (launch initial) lastInv
  change Safety.traceSafeList _ _ _ _
  rw [Program.backward_split]
  simp only [List.append_assoc,List.singleton_append]
  rw [Safety.traceSafeList_append]
  refine ⟨backward_prefix_safe a s b p.plans seed final initial hscale bounds hb,?_⟩
  intro u hu
  rw [he] at hu
  cases Option.some.inj hu
  apply Safety.traceSafeList_cons_of_step
  · rw [Safety.stepSafe_unfold]
    exact first_range_safe a s b p.plans seed final initial entry (launch initial)
      (block_positive s shape).2 p.dq.covers inv bounds hb
  · simpa only [step] using ran
  · apply (Safety.traceSafeList_append (Staged.algebra a p.plans) bounds
      (Program.backwardSecond (layout s b) backwardRegions seed final).before
      [.forRange "_t" 0 (layout s b).T 1 (Program.backwardSecond (layout s b) backwardRegions seed final).body]
      (backwardFirstTail a s b p.plans initial seed (launch initial) 0 entry)).mpr
    refine ⟨backward_middle_safe a s b p.plans seed final initial _ (launch initial) lastInv bounds hb,?_⟩
    intro v hv
    rw [hm] at hv
    cases Option.some.inj hv
    apply Safety.traceSafeList_cons_intro
    · rw [Safety.stepSafe_unfold]
      exact second_range_safe a s b p.plans seed final initial middle (launch initial)
        (block_positive s shape).1 (block_positive s shape).2 p.dk.covers p.dv.covers secondInv bounds hb
    · intro t _
      exact Safety.traceSafeList_nil _ _ _

#axiomsClean forward_safe
#axiomsClean backward_safe
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.SourceSafety
