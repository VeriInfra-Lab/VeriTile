import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.TransportBody

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.MemoryTransport
open VeriTile.Triton FP.Structural FPProgram FPValues FPSetup FPBody FPInvariant
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

/-- Reuse the original constructed iterations and ordered recurrence. The
arbitrary backing memory is unchanged throughout the read-only source range. -/
theorem range_transport (M : Algebra α) (P : Primitives α) (h : Call)
    (batch split slot c : Nat) (data : Data α) (s : State α) (hc : c ≤ h.cycles)
    (ready : Invariant h data batch split slot (accumulator M P h data batch slot split c) s)
    (memory : RegionName → Nat → Cell α)
    (inputRead : ∀ k, k < h.K → memory "X" (inputOffset h batch k) =
      .mk (dtype h.storage).toTileDType (ofFloat _ (data.input (inputOffset h batch k))))
    (weightRead : ∀ n k, n < h.N → k < h.K → memory "W" (weightOffset h slot n k) =
      .mk (dtype h.storage).toTileDType (ofFloat _ (data.weights (weightOffset h slot n k)))) :
    ∃ t, rangeInt (machine M P) "k" ((c*16384:Nat):Int) (h.K:Int) 16384 (body h) s = some t ∧
      Invariant h data batch split slot (accumulator M P h data batch slot split h.cycles) t ∧
      rangeInt (machine M P) "k" ((c*16384:Nat):Int) (h.K:Int) 16384 (body h)
        (rebase s memory) = some (rebase t memory) := by
  by_cases more : c < h.cycles
  · have active : ((c*16384:Nat):Int) < (h.K:Int) := by
      unfold VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Call.cycles at more
      omega
    let next := iteration M P h batch split slot c data (accumulator M P h data batch slot split c) s ready
    have nextReady : Invariant h data batch split slot
        (accumulator M P h data batch slot split (c+1)) next.val :=
      iteration_ready M P h batch split slot c data _ s ready
    obtain ⟨t, oldRun, nextFinal, actualRun⟩ := range_transport M P h batch split slot (c+1)
      data next.val (by omega) nextReady memory inputRead weightRead
    refine ⟨t, ?_, nextFinal, ?_⟩
    · rw [rangeInt]
      simp only [show (16384:Nat) ≠ 0 from by decide, if_false, active, if_true,
        next.property, Option.bind_some]
      simpa only [Nat.add_mul, Nat.one_mul, Int.natCast_add, Int.natCast_mul] using oldRun
    · rw [rangeInt]
      simp only [show (16384:Nat) ≠ 0 from by decide, if_false, active, if_true,
        body_transport M P h batch split slot c data _ s ready memory inputRead weightRead,
        Option.bind_some]
      simpa only [Nat.add_mul, Nat.one_mul, Int.natCast_add, Int.natCast_mul] using actualRun
  · have equal : c = h.cycles := by omega
    have done : ¬ ((c*16384:Nat):Int) < (h.K:Int) := by
      rw [equal]
      have := VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.covered h
      omega
    refine ⟨s, ?_, equal ▸ ready, ?_⟩
    all_goals
      rw [rangeInt]
      simp only [show (16384:Nat) ≠ 0 from by decide, if_false, done]
termination_by h.cycles-c

theorem loop_transport (M : Algebra α) (P : Primitives α) (h : Call)
    (batch split slot : Nat) (data : Data α) (metadata : data.indices batch = (slot:Int))
    (memory : RegionName → Nat → Cell α)
    (inputRead : ∀ k, k < h.K → memory "X" (inputOffset h batch k) =
      .mk (dtype h.storage).toTileDType (ofFloat _ (data.input (inputOffset h batch k))))
    (weightRead : ∀ n k, n < h.N → k < h.K → memory "W" (weightOffset h slot n k) =
      .mk (dtype h.storage).toTileDType (ofFloat _ (data.weights (weightOffset h slot n k)))) :
    ∃ t, step (machine M P) (loop h) (preloop M P h batch split slot data metadata).val = some t ∧
      Invariant h data batch split slot (accumulator M P h data batch slot split h.cycles) t ∧
      step (machine M P) (loop h) (rebase (preloop M P h batch split slot data metadata).val memory) =
        some (rebase t memory) := by
  simp only [FPProgram.loop,(SourceControl.constant_stride (machine M P) h.K (body h)).step]
  simpa only [step, Nat.zero_mul, evalOp_unfold, Option.bind_some, Int.natCast_zero]
    using range_transport M P h batch split slot 0 data _ (Nat.zero_le _)
      (setup_ready M P h batch split slot data metadata) memory inputRead weightRead

#axiomsClean range_transport
#axiomsClean loop_transport
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.MemoryTransport
