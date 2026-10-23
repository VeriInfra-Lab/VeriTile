import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeSafetyDiagonal

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite FPDiagonal
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 400000
set_option maxRecDepth 4000

theorem diagonal_range (M : Algebra α) (split : Bool) (c : Layout) (scale : ℝ)
    (bounds : RegionName → Nat) (initial current : State α)
    (ctx : Context M split c initial current 0) (capacity : DiagonalCapacity split c initial bounds) :
    Safety.rangeSafe (engine M) bounds "j" 0 (columns split c initial) 1
      (parts split c scale).body current := by
  apply Safety.rangeSafe_of_invariant (engine M) bounds "j" (columns split c initial) 1
    (parts split c scale).body (fun j t => Context M split c initial t j)
  · intro j active t h
    exact diagonal_body M split c scale bounds initial t j active h capacity
  · intro j active t h u ran
    have same := Option.some.inj (ran.symm.trans (bodyExecution M split c scale initial t j h).property)
    subst u
    exact bodyContext M split c scale initial t j h
  · exact ctx

#axiomsClean diagonal_range

private theorem diagonal_condition_safe (M : Algebra α) (split : Bool) (c : Layout) (scale : ℝ)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.safeOp (engine M) none bounds s (branch split c scale).condition := by
  cases split <;>
    simp [branch,kernel,TypedIntra.original,TypedIntra.source,TypedSplit.original,TypedSplit.source,
      ComputeKernel.surfaceBody,Source.conditional?]
  all_goals simp [Safety.safeOp_unfold]

private theorem diagonal_stop_safe (M : Algebra α) (split : Bool) (c : Layout) (scale : ℝ)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.safeOp (engine M) none bounds s (parts split c scale).stop := by
  cases split <;>
    simp [parts,branch,kernel,TypedIntra.original,TypedIntra.source,TypedSplit.original,TypedSplit.source,
      ComputeKernel.surfaceBody,Source.conditional?,Source.dynamicRange?]
  all_goals simp [Safety.safeOp_unfold]

theorem diagonal_source (M : Algebra α) (split : Bool) (c : Layout) (scale : ℝ)
    (bounds : RegionName → Nat) (s : State α) (capacity : DiagonalCapacity split c s bounds) :
    TraceSafe (engine M) bounds (kernel split c scale) s := by
  let start := entryExecution M split c scale s
  have hs : run (engine M) (parts split c scale).before (prologue M split c scale s).val=some start.val := by
    simpa only [run_append,(prologue M split c scale s).property,Option.bind_some] using start.property
  have prefixSafe := (Safety.traceSafeList_append (engine M) bounds
    (branch split c scale).before (parts split c scale).before s).mp
      (diagonal_entry M split c scale bounds s capacity)
  change Safety.traceSafeList (engine M) bounds (kernel split c scale).surfaceBody s
  rw [source_parts,Safety.traceSafeList_append]
  refine ⟨prefixSafe.1,?_⟩
  intro t ran
  have same := Option.some.inj (ran.symm.trans (prologue M split c scale s).property)
  subst t
  rw [singleton,Safety.stepSafe_unfold]
  constructor
  · exact diagonal_condition_safe M split c scale bounds _
  · rw [condition_eval]
    by_cases active : enabled split c s
    · simp only [active,decide_true,ite_true]
      rw [Safety.traceSafeList_append]
      refine ⟨prefixSafe.2 _ (prologue M split c scale s).property,?_⟩
      intro t ran
      have same := Option.some.inj (ran.symm.trans hs)
      subst t
      rw [singleton,Safety.stepSafe_unfold]
      have h0 : (parts split c scale).start = .constNat 0 := by cases split <;> rfl
      have h1 : (parts split c scale).stride = .constNat 1 := by cases split <;> rfl
      have hstop := stop_eval M split c scale s
      refine ⟨?_,?_,?_,?_⟩
      · simp [h0,Safety.safeOp_unfold]
      · exact diagonal_stop_safe M split c scale bounds _
      · simp [h1,Safety.safeOp_unfold]
      · rw [h0,h1,hstop]
        simp only [evalOp_unfold]
        exact diagonal_range M split c scale bounds s start.val
          (entryContext M split c scale s) capacity
    · simp only [active,decide_false,Bool.false_eq_true,ite_false]

#axiomsClean diagonal_source
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeSafety
