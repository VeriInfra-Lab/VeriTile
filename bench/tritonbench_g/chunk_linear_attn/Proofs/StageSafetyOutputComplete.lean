import bench.tritonbench_g.chunk_linear_attn.Proofs.StageSafetyOutput

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.StageSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 3000000
variable {α : Type} [Inhabited α]

theorem output_range_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (bounds : RegionName → Nat) (pid : PID) (input : Memory α) (s : State α)
    (ctx : OutputReady p pid input s) (a : OutputAcc p α) (ha : HasOutput p a s)
    (capacity : OutputCapacity p bounds) (head : pid.z < p.B * p.H) (time : pid.y < p.NT) :
    Safety.rangeSafe (engine (withDot M D)) bounds "i_k" 0 (keyBlocks p) 1
      (outputParts p scale).body s := by
  let invariant := fun (_ : Nat) (t : State α) =>
    OutputReady p pid input t ∧ ∃ values, HasOutput p values t
  apply Safety.rangeSafe_of_invariant _ _ _ _ _ _ invariant
  · intro n _ t ⟨ready, values, available⟩
    exact output_body_safe M D p scale bounds pid input t ready values available n capacity head time
  · intro n _ t ⟨ready, values, available⟩ u ran
    let actual := outputBody M D p scale pid input t n values ready available
    have same : actual.val = u := Option.some.inj (actual.property.symm.trans ran)
    subst u
    have next := outputBody_ready M D p scale pid input t n values ready available
    exact ⟨next.1, _, next.2⟩
  · exact ⟨ctx, a, ha⟩

theorem output_source_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (bounds : RegionName → Nat) (s : State α) (capacity : OutputCapacity p bounds)
    (head : s.pids 2 < p.B * p.H) (time : s.pids 1 < p.NT) :
    TraceSafe (engine (withDot M D)) bounds (forwardO p scale) s := by
  let pid : PID := ⟨s.pids 0,s.pids 1,s.pids 2⟩
  let start := outputEntry M D p scale s
  have ready := outputEntry_ready M D p scale s
  obtain ⟨mid, loopRan, midReady, midValues⟩ :=
    outputLoop M D p scale pid s.mem start.val ready.1 ready.2
  have h0 : (outputParts p scale).start = .constNat 0 := rfl
  have hn : (outputParts p scale).stop = .div .nat .nil
      (.sub .nat .nil (.add .nat .nil (.constNat p.K) (.constNat p.BK)) (.constNat 1))
      (.constNat p.BK) := rfl
  have h1 : (outputParts p scale).stride = .constNat 1 := rfl
  let loop := ComputeStmt.forRangeDyn "i_k" (outputParts p scale).start (outputParts p scale).stop
    (outputParts p scale).stride (outputParts p scale).body
  have loopStep : step (engine (withDot M D)) loop start.val = some mid := by
    simpa [loop,step,h0,hn,h1,evalOp_unfold,Option.bind,numeric,bop,keyBlocks] using loopRan
  have loopSafe : Safety.stepSafe (engine (withDot M D)) bounds loop start.val := by
    have safe := output_range_safe M D p scale bounds pid s.mem start.val ready.1 _ ready.2
      capacity head time
    simpa [loop,Safety.stepSafe_unfold,Safety.safeOp_unfold,h0,hn,h1,evalOp_unfold,
      Option.bind,numeric,bop,keyBlocks] using safe
  unfold TraceSafe
  rw [output_parts, List.append_assoc, Safety.traceSafeList_append]
  refine ⟨output_entry_safe M D p scale bounds s, ?_⟩
  intro t ran
  have same : start.val = t := Option.some.inj (start.property.symm.trans ran)
  subst t
  apply Safety.traceSafeList_cons_of_step loopSafe loopStep
  exact output_finish_safe M D p scale bounds pid s.mem mid midReady _ midValues capacity head

#axiomsClean output_range_safe
#axiomsClean output_source_safe
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.StageSafety
