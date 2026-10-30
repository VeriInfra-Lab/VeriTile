import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.AccessSafety

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.AccessSafety
open VeriTile.Triton FP.Structural FPProgram FPValues FPSetup FPBody
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

theorem range_safe (M : Algebra α) (P : Primitives α) (h : Call)
    (batch split slot c : Nat) (data : Data α) (s : State α) (hc : c ≤ h.cycles)
    (ready : Invariant h data batch split slot (accumulator M P h data batch slot split c) s)
    (bounds : RegionName → Nat)
    (inputBounds : ∀ k, k < h.K → inputOffset h batch k < bounds "X")
    (weightBounds : ∀ n k, n < h.N → k < h.K → weightOffset h slot n k < bounds "W") :
    Safety.rangeIntSafe (machine M P) bounds "k" ((c*16384:Nat):Int) (h.K:Int) 16384 (body h) s := by
  rw [Safety.rangeIntSafe]
  simp only [show (16384:Nat) ≠ 0 from by decide, if_false]
  by_cases more : c < h.cycles
  · have active : ((c*16384:Nat):Int) < (h.K:Int) := by
      unfold VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Call.cycles at more
      omega
    rw [if_pos active]
    refine ⟨body_safe M P h batch split slot c data _ s ready bounds inputBounds weightBounds, ?_⟩
    rw [(iteration M P h batch split slot c data _ s ready).property]
    have next := FPInvariant.iteration_ready M P h batch split slot c data _ s ready
    have nextReady : Invariant h data batch split slot
        (accumulator M P h data batch slot split (c+1))
        (iteration M P h batch split slot c data _ s ready).val := next
    simpa only [Nat.add_mul,Nat.one_mul,Int.natCast_add] using
      range_safe M P h batch split slot (c+1) data _ (by omega) nextReady bounds inputBounds weightBounds
  · have equal : c=h.cycles := by omega
    have done : ¬((c*16384:Nat):Int) < (h.K:Int) := by
      rw [equal]
      have := VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.covered h
      omega
    simp only [done,if_false]
termination_by h.cycles-c

#axiomsClean range_safe

/-- Every source access is inside its declared logical window. The sentinel
still reads metadata; active programs read X even when there are no output rows. -/
theorem source_safe (M : Algebra α) (P : Primitives α) (h : Call)
    (batch split : Nat) (data : Data α)
    (valid : data.indices batch = -1 ∨ 0 ≤ data.indices batch)
    (bounds : RegionName → Nat) (indexBound : batch < bounds "Indices")
    (inputBounds : data.indices batch ≠ -1 → ∀ k, k < h.K → inputOffset h batch k < bounds "X")
    (weightBounds : data.indices batch ≠ -1 → ∀ n k, n < h.N → k < h.K →
      weightOffset h (data.indices batch).toNat n k < bounds "W")
    (outputBounds : data.indices batch ≠ -1 → ∀ n, n < h.N → outputOffset h batch n < bounds "Y") :
    TraceSafe (machine M P) bounds (Surface.source h) (entry h batch split data) := by
  change Safety.traceSafeList (machine M P) bounds (code h) (entry h batch split data)
  rw [complete_code, Safety.traceSafeList_append]
  refine ⟨prefix_safe M P h batch split data bounds indexBound, ?_⟩
  intro initial ran
  have same : initial = (preguard M P h batch split data).val :=
    Option.some.inj (ran.symm.trans (preguard M P h batch split data).property)
  subst initial
  apply Safety.traceSafeList_cons_intro
  · rw [Safety.stepSafe_unfold]
    constructor
    · simp only [FPProgram.guard,Safety.safeOp_unfold,and_self]
    · rw [FPExecution.condition]
      change (if decide (data.indices batch ≠ -1) = Bool.true then
        Safety.traceSafeList (machine M P) bounds (active h) (preguard M P h batch split data).val else True)
      by_cases sentinel : data.indices batch = -1
      · simp only [sentinel,ne_self_iff_false,decide_false,Bool.false_eq_true,if_false]
      · simp only [decide_eq_true_eq,if_pos sentinel]
        have nonnegative := valid.resolve_left sentinel
        have metadata : data.indices batch = ((data.indices batch).toNat:Int) :=
          (Int.toNat_of_nonneg nonnegative).symm
        rw [complete_active, Safety.traceSafeList_append]
        refine ⟨setup_safe M P h batch split _ data metadata bounds, ?_⟩
        intro start startRun
        have same : start = (preloop M P h batch split _ data metadata).val :=
          Option.some.inj (startRun.symm.trans (preloop M P h batch split _ data metadata).property)
        subst start
        apply Safety.traceSafeList_cons_intro
        · apply ((SourceControl.constant_stride (machine M P) h.K (body h)).safe bounds _).2
          simpa only [Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,
            Nat.zero_mul,Int.natCast_zero,true_and] using
            range_safe M P h batch split _ 0 data _ (Nat.zero_le _)
              (FPInvariant.setup_ready M P h batch split _ data metadata) bounds
              (inputBounds sentinel) (weightBounds sentinel)
        · intro after afterRun
          obtain ⟨last,lastRun,lastReady⟩ := FPLoop.loop_runs M P h batch split _ data metadata
          have same : after = last := Option.some.inj (afterRun.symm.trans lastRun)
          subst after
          exact tail_safe M P h batch split _ data _ last lastReady bounds (outputBounds sentinel)
  · intro last _
    exact Safety.traceSafeList_nil _ _ last

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.AccessSafety
