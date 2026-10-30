import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.RebasedSafety
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.TransportLoop

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.RebasedSafety
open VeriTile.Triton FP.Structural FPProgram FPValues FPSetup FPBody MemoryTransport
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

theorem range_safe (M : Algebra α) (P : Primitives α) (h : Call)
    (batch split slot c : Nat) (data : Data α) (s : State α) (hc : c ≤ h.cycles)
    (ready : Invariant h data batch split slot (accumulator M P h data batch slot split c) s)
    (memory : RegionName → Nat → Cell α)
    (inputRead : ∀ k, k < h.K → memory "X" (inputOffset h batch k) =
      .mk (dtype h.storage).toTileDType (ofFloat _ (data.input (inputOffset h batch k))))
    (weightRead : ∀ n k, n < h.N → k < h.K → memory "W" (weightOffset h slot n k) =
      .mk (dtype h.storage).toTileDType (ofFloat _ (data.weights (weightOffset h slot n k))))
    (bounds : RegionName → Nat)
    (inputBounds : ∀ k, k < h.K → inputOffset h batch k < bounds "X")
    (weightBounds : ∀ n k, n < h.N → k < h.K → weightOffset h slot n k < bounds "W") :
    Safety.rangeIntSafe (machine M P) bounds "k" ((c*16384:Nat):Int) (h.K:Int) 16384
      (FPProgram.body h) (rebase s memory) := by
  rw [Safety.rangeIntSafe]
  simp only [show (16384:Nat) ≠ 0 from by decide,if_false]
  by_cases more : c < h.cycles
  · have active : ((c*16384:Nat):Int) < (h.K:Int) := by
      unfold VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Call.cycles at more
      omega
    rw [if_pos active]
    refine ⟨body_safe M P h batch split slot c data _ s ready memory bounds inputBounds weightBounds,?_⟩
    rw [MemoryTransport.body_transport M P h batch split slot c data _ s ready memory inputRead weightRead]
    have nextReady : Invariant h data batch split slot
        (accumulator M P h data batch slot split (c+1))
        (iteration M P h batch split slot c data _ s ready).val :=
      FPInvariant.iteration_ready M P h batch split slot c data _ s ready
    simpa only [Nat.add_mul,Nat.one_mul,Int.natCast_add] using
      range_safe M P h batch split slot (c+1) data _ (by omega) nextReady memory inputRead weightRead
        bounds inputBounds weightBounds
  · have equal : c=h.cycles := by omega
    have done : ¬((c*16384:Nat):Int) < (h.K:Int) := by
      rw [equal]
      have := VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.covered h
      omega
    simp only [done,if_false]
termination_by h.cycles-c

/-- Actual source safety on arbitrary backing memory uses the same masked
read cells as the existing execution, and preserves all exterior tags. -/
theorem source_safe (M : Algebra α) (P : Primitives α) (h : Call)
    (batch split : Nat) (data : Data α) (memory : RegionName → Nat → Cell α)
    (valid : data.indices batch=-1 ∨ 0 ≤ data.indices batch)
    (index : memory "Indices" batch = .mk .int (data.indices batch))
    (inputRead : data.indices batch ≠ -1 → ∀ k, k < h.K → memory "X" (inputOffset h batch k) =
      .mk (dtype h.storage).toTileDType (ofFloat _ (data.input (inputOffset h batch k))))
    (weightRead : data.indices batch ≠ -1 → ∀ n k, n < h.N → k < h.K →
      memory "W" (weightOffset h (data.indices batch).toNat n k) =
        .mk (dtype h.storage).toTileDType (ofFloat _ (data.weights (weightOffset h (data.indices batch).toNat n k))))
    (bounds : RegionName → Nat) (indexBound : batch < bounds "Indices")
    (inputBounds : data.indices batch ≠ -1 → ∀ k, k < h.K → inputOffset h batch k < bounds "X")
    (weightBounds : data.indices batch ≠ -1 → ∀ n k, n < h.N → k < h.K →
      weightOffset h (data.indices batch).toNat n k < bounds "W")
    (outputBounds : data.indices batch ≠ -1 → ∀ n, n < h.N → outputOffset h batch n < bounds "Y") :
    TraceSafe (machine M P) bounds (Surface.source h) (rebase (entry h batch split data) memory) := by
  have condition : evalOp (machine M P) none FPProgram.guard
      (rebase (preguard M P h batch split data).val memory) =
      some (fun _ => decide (data.indices batch ≠ -1)) := FPExecution.condition M P h batch split data
  change Safety.traceSafeList (machine M P) bounds (FPProgram.code h) (rebase (entry h batch split data) memory)
  rw [complete_code,Safety.traceSafeList_append]
  refine ⟨prefix_safe M P h batch split data memory bounds indexBound,?_⟩
  intro initial ran
  have same : initial = rebase (preguard M P h batch split data).val memory :=
    Option.some.inj (ran.symm.trans (prefix_transport M P h batch split data memory index))
  subst initial
  apply Safety.traceSafeList_cons_intro
  · rw [Safety.stepSafe_unfold]
    constructor
    · simp only [FPProgram.guard,Safety.safeOp_unfold,and_self]
    · rw [condition]
      change (if decide (data.indices batch ≠ -1) = Bool.true then
        Safety.traceSafeList (machine M P) bounds (FPProgram.active h)
          (rebase (preguard M P h batch split data).val memory) else True)
      by_cases sentinel : data.indices batch = -1
      · simp only [sentinel,ne_self_iff_false,decide_false,Bool.false_eq_true,if_false]
      · simp only [decide_eq_true_eq,if_pos sentinel]
        have metadata : data.indices batch = ((data.indices batch).toNat:Int) :=
          (Int.toNat_of_nonneg (valid.resolve_left sentinel)).symm
        rw [complete_active,Safety.traceSafeList_append]
        refine ⟨setup_safe M P h batch split _ data metadata memory bounds,?_⟩
        intro start startRun
        have same : start = rebase (preloop M P h batch split _ data metadata).val memory :=
          Option.some.inj (startRun.symm.trans (MemoryTransport.setup M P h batch split _ data metadata memory))
        subst start
        apply Safety.traceSafeList_cons_intro
        · apply ((SourceControl.constant_stride (machine M P) h.K (body h)).safe bounds _).2
          simpa only [Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,
            Nat.zero_mul,Int.natCast_zero,true_and] using
            range_safe M P h batch split _ 0 data _ (Nat.zero_le _)
              (FPInvariant.setup_ready M P h batch split _ data metadata) memory
              (inputRead sentinel) (weightRead sentinel) bounds (inputBounds sentinel) (weightBounds sentinel)
        · intro after afterRun
          obtain ⟨last,_,lastReady,lastRun⟩ := loop_transport M P h batch split _ data metadata memory
            (inputRead sentinel) (weightRead sentinel)
          have same : after = rebase last memory := Option.some.inj (afterRun.symm.trans lastRun)
          subst after
          exact tail_safe M P h batch split _ data _ last lastReady memory bounds (outputBounds sentinel)
  · intro last _
    exact Safety.traceSafeList_nil _ _ last

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.RebasedSafety
