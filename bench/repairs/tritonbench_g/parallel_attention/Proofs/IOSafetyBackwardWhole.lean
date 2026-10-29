import bench.repairs.tritonbench_g.parallel_attention.Proofs.IOSafetyBackward

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.IOSafety
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem backward_first_range (M : Algebra α) (D : DotPrimitive α) (c : BackwardConfig)
    (bounds : RegionName → Nat) (s t : State α) (start stop : Nat)
    (acc : BackwardAccumulator α c.BTL c.BK c.BV) (h : BackwardReady M c s t acc)
    (hb : BackwardBounds c bounds) :
    Safety.rangeSafe (withDot M D) bounds "j" start stop 1 (backwardFirstBody c) t := by
  apply Safety.rangeSafe_of_invariant (withDot M D) bounds "j" stop 1 _
    (fun _ u=>∃ a,BackwardReady M c s u a)
  · rintro q _ u ⟨a,hu⟩
    exact backward_first_body M D c bounds _ q (by simp [State.setReg]) hb
  · rintro q _ u ⟨a,hu⟩ v ran
    obtain ⟨w,hw,hready⟩:=backward_first_body_run M D c s _ q a
      (backwardReady_j M c s u a hu q) (by simp [State.setReg])
    rw [hw] at ran
    cases Option.some.inj ran
    exact ⟨_,hready⟩
  · exact ⟨acc,h⟩

theorem backward_second_range (M : Algebra α) (D : DotPrimitive α) (c : BackwardConfig)
    (bounds : RegionName → Nat) (s t : State α) (start stop : Nat)
    (acc : BackwardAccumulator α c.BTL c.BK c.BV) (h : BackwardDiagonalReady M c s t start acc)
    (hb : BackwardBounds c bounds) :
    Safety.rangeSafe (withDot M D) bounds "i" start stop c.BTS (backwardSecondBody c) t := by
  apply Safety.rangeSafe_of_invariant (withDot M D) bounds "i" stop c.BTS _
    (fun q u=>∃ a,BackwardDiagonalReady M c s u q a)
  · rintro q _ u ⟨a,hu⟩
    exact backward_second_body M D c bounds _ q (by simp [State.setReg]) hb
  · rintro q _ u ⟨a,hu⟩ v ran
    obtain ⟨w,hw,hready⟩:=backward_second_body_run M D c s _ q a
      (backwardDiagonalReady_i M c s u q a hu q) (by simp [State.setReg])
    rw [hw] at ran
    cases Option.some.inj ran
    exact ⟨_,hready⟩
  · exact ⟨acc,h⟩

theorem backward_source (M : Algebra α) (D : DotPrimitive α) (c : BackwardConfig)
    (bounds : RegionName → Nat) (s : State α) (hb : BackwardBounds c bounds) :
    TraceSafe (withDot M D) bounds (backwardKernel c) s := by
  obtain ⟨initial,hp,hi⟩:=backward_prefix_run M D c s
  obtain ⟨first,hr,hf⟩:=backward_first_range_run M D c s initial 0 (descendingCount c) _ hi
  obtain ⟨diagonal,hm,hd⟩:=backward_middle_run M D c s first _ hf
  obtain ⟨second,hs,hz⟩:=backward_second_range_run M D c s diagonal
    (c.i_c*c.BTL) ((c.i_c+1)*c.BTL) _ hd
  change Safety.traceSafeList _ _ _ _
  rw [backward_source_split]
  simp only [List.append_assoc]
  rw [Safety.traceSafeList_append]
  refine ⟨backward_prefix M D c bounds s hb,?_⟩
  intro t ht
  rw [hp] at ht
  cases Option.some.inj ht
  simp only [List.singleton_append]
  apply Safety.traceSafeList_cons_of_step
  · have safe:=backward_first_range M D c bounds s initial 0 (descendingCount c) _ hi hb
    simpa [backwardFirstRange,backwardFirstBody,backwardKernel,backwardSource,ComputeKernel.surfaceBody,
      Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,numeric,bop,Option.bind,descendingCount] using safe
  · rw [backward_first_range_statement]
    exact hr
  · rw [Safety.traceSafeList_append]
    refine ⟨backward_middle M D c bounds first,?_⟩
    intro t ht
    rw [hm] at ht
    cases Option.some.inj ht
    apply Safety.traceSafeList_cons_of_step
    · have safe:=backward_second_range M D c bounds s diagonal
        (c.i_c*c.BTL) ((c.i_c+1)*c.BTL) _ hd hb
      simpa [backwardSecondRange,backwardSecondBody,backwardKernel,backwardSource,ComputeKernel.surfaceBody,
        Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,numeric,bop,Option.bind] using safe
    · rw [backward_second_range_statement]
      exact hs
    · exact backward_suffix M D c bounds s second _ hz hb

#axiomsClean backward_source
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.IOSafety
