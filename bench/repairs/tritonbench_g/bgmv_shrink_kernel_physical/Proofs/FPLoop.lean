import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPInvariant
namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPLoop
open VeriTile.Triton FP.Structural FPProgram VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues FPSetup FPBody FPInvariant
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

/-- The complete arbitrary-length source range, with the exact ordered FP
accumulator recurrence. This includes zero iterations and every K tail. -/
theorem range_runs (M : Algebra α) (P : Primitives α) (h : Call)
    (batch split slot c : Nat) (data : Data α) (s : State α) (hc : c ≤ h.cycles)
    (ready : Invariant h data batch split slot (accumulator M P h data batch slot split c) s) :
    ∃ t,FP.Structural.rangeInt (machine M P) "k" ((c*16384:Nat):Int) (h.K:Int) 16384 (body h) s=some t ∧
      Invariant h data batch split slot (accumulator M P h data batch slot split h.cycles) t := by
  by_cases more : c < h.cycles
  · have active : ((c*16384:Nat):Int) < (h.K:Int) := by
      unfold VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Call.cycles at more
      omega
    let next := iteration M P h batch split slot c data (accumulator M P h data batch slot split c) s ready
    have nextReady : Invariant h data batch split slot
        (accumulator M P h data batch slot split (c+1)) next.val :=
      iteration_ready M P h batch split slot c data _ s ready
    obtain ⟨t,he,hi⟩ := range_runs M P h batch split slot (c+1) data next.val (by omega) nextReady
    refine ⟨t,?_,hi⟩
    rw [FP.Structural.rangeInt]
    simp only [show (16384:Nat) ≠ 0 from by decide,if_false,active,if_true,next.property,Option.bind_some]
    simpa only [Nat.add_mul,Nat.one_mul,Int.natCast_add,Int.natCast_mul] using he
  · have equal : c=h.cycles := by omega
    have done : ¬((c*16384:Nat):Int) < (h.K:Int) := by
      rw [equal]
      have := VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.covered h
      omega
    refine ⟨s,?_,equal ▸ ready⟩
    rw [FP.Structural.rangeInt]
    simp only [show (16384:Nat) ≠ 0 from by decide,if_false,done]
termination_by h.cycles-c

theorem loop_runs (M : Algebra α) (P : Primitives α) (h : Call)
    (batch split slot : Nat) (data : Data α) (metadata : data.indices batch=(slot:Int)) :
    ∃ t,step (machine M P) (loop h) (preloop M P h batch split slot data metadata).val=some t ∧
      Invariant h data batch split slot (accumulator M P h data batch slot split h.cycles) t := by
  simpa only [FPProgram.loop,step,Nat.zero_mul,evalOp_unfold,Option.bind_some,Int.natCast_zero] using range_runs M P h batch split slot 0 data _ (Nat.zero_le _)
    (setup_ready M P h batch split slot data metadata)

#axiomsClean range_runs
#axiomsClean loop_runs
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPLoop
