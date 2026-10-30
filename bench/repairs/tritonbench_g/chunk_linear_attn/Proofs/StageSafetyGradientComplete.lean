import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.StageSafetyGradient

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.StageSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 3000000
variable {α : Type} [Inhabited α]

theorem gradient_range_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (bounds : RegionName → Nat) (pid : PID) (input : Memory α) (s : State α)
    (ctx : GradientReady p pid (p.B * p.H) input s)
    (fixed : GradientFixed M D p pid input scale s) (a : GradAcc p α) (ha : HasGrad p a s)
    (capacity : GradientCapacity p bounds) (head : pid.z < p.B * p.H)
    (keyBlock : pid.x < keyBlocks p) :
    Safety.rangeSafe (engine (withDot M D)) bounds "i_v" 0 (valueBlocks p) 1
      (gradientParts p scale).body s := by
  let invariant := fun (_ : Nat) (t : State α) =>
    GradientReady p pid (p.B * p.H) input t ∧ GradientFixed M D p pid input scale t ∧
      ∃ values, HasGrad p values t
  apply Safety.rangeSafe_of_invariant _ _ _ _ _ _ invariant
  · intro n _ t ⟨ready, fixed, values, available⟩
    exact gradient_body_safe M D p scale bounds pid input t ready fixed values available n
      capacity head keyBlock
  · intro n _ t ⟨ready, fixed, values, available⟩ u ran
    let actual := gradientBody M D p scale pid (p.B * p.H) input t n values ready fixed available
    have same : actual.val = u := Option.some.inj (actual.property.symm.trans ran)
    subst u
    have next := gradientBody_values M D p scale pid (p.B * p.H) input t n values ready fixed available
    exact ⟨next.1, next.2.1, _, next.2.2⟩
  · exact ⟨ctx, fixed, a, ha⟩

theorem gradient_source_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (bounds : RegionName → Nat) (s : State α) (capacity : GradientCapacity p bounds)
    (head : s.pids 2 < p.B * p.H) (keyBlock : s.pids 0 < keyBlocks p)
    (heads : s.numPids 2 = p.B * p.H) :
    TraceSafe (engine (withDot M D)) bounds (backwardQKV p scale) s := by
  let pid : PID := ⟨s.pids 0,s.pids 1,s.pids 2⟩
  let start := gradientEntry M D p scale s
  have ready := gradientEntry_values M D p scale s
  have ctx : GradientReady p pid (p.B * p.H) s.mem start.val := by
    simpa only [heads] using ready.1
  obtain ⟨mid, loopRan, midReady, midFixed, midValues, _⟩ :=
    gradientLoop M D p scale pid (p.B * p.H) s.mem start.val ctx ready.2.1 ready.2.2.1
  have h0 : (gradientParts p scale).start = .constNat 0 := rfl
  have hn : (gradientParts p scale).stop = .div .nat .nil
      (.sub .nat .nil (.add .nat .nil (.constNat p.V) (.constNat p.BV)) (.constNat 1))
      (.constNat p.BV) := rfl
  have h1 : (gradientParts p scale).stride = .constNat 1 := rfl
  let loop := ComputeStmt.forRangeDyn "i_v" (gradientParts p scale).start (gradientParts p scale).stop
    (gradientParts p scale).stride (gradientParts p scale).body
  have loopStep : step (engine (withDot M D)) loop start.val = some mid := by
    simpa [loop,step,h0,hn,h1,evalOp_unfold,Option.bind,numeric,bop,valueBlocks] using loopRan
  have loopSafe : Safety.stepSafe (engine (withDot M D)) bounds loop start.val := by
    have safe := gradient_range_safe M D p scale bounds pid s.mem start.val ctx ready.2.1
      _ ready.2.2.1 capacity head keyBlock
    simpa [loop,Safety.stepSafe_unfold,Safety.safeOp_unfold,h0,hn,h1,evalOp_unfold,
      Option.bind,numeric,bop,valueBlocks] using safe
  unfold TraceSafe
  rw [gradient_parts, List.append_assoc, Safety.traceSafeList_append]
  refine ⟨gradient_entry_safe M D p scale bounds s capacity head, ?_⟩
  intro t ran
  have same : start.val = t := Option.some.inj (start.property.symm.trans ran)
  subst t
  apply Safety.traceSafeList_cons_of_step loopSafe loopStep
  exact gradient_finish_safe M D p scale bounds pid (p.B * p.H) s.mem mid midReady midFixed
    _ midValues capacity head

#axiomsClean gradient_range_safe
#axiomsClean gradient_source_safe
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.StageSafety
