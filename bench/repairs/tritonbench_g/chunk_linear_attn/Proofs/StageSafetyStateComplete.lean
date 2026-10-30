import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.StageSafetyState

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.StageSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 3000000
variable {α : Type} [Inhabited α]

theorem state_body_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (rev seed final : Bool) (bounds : RegionName → Nat) (pid : PID)
    (input : Memory α) (s : State α) (ctx : StateReady pid input s)
    (h : HValues p α) (carry : HasCarry p rev h s)
    (n : Nat) (active : n < p.NT) (capacity : StateCapacity p rev seed final bounds)
    (head : pid.z < p.B * p.H) :
    Safety.traceSafeList (engine (withDot M D)) bounds
      (stateParts p scale rev seed final).body
      (s.setReg (loopName rev) .nat [] (fun _ => n)) := by
  have split : (stateParts p scale rev seed final).body =
      stateStorePrefix p scale rev seed final ++ stateUpdateSuffix p scale rev seed final :=
    (List.take_append_drop _ _).symm
  rw [split, Safety.traceSafeList_append]
  refine ⟨state_store_safe M D p scale rev seed final bounds pid input s ctx n active capacity head, ?_⟩
  intro t ran
  let start := stateStoreExecution M D p scale rev seed final pid input s n h ctx carry
  have same : start.val = t := Option.some.inj (start.property.symm.trans ran)
  subst t
  have ready := stateStore_ready M D p scale rev seed final pid input s n h ctx carry
  have pointers := stateStore_pointers M D p scale rev seed final pid input s n h ctx carry
  exact state_update_safe M D p scale rev seed final bounds pid input start.val ready.1 n
    pointers capacity head

theorem state_loop_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (rev seed final : Bool) (bounds : RegionName → Nat) (pid : PID)
    (input : Memory α) (s : State α) (ctx : StateReady pid input s)
    (h : HValues p α) (carry : HasCarry p rev h s)
    (capacity : StateCapacity p rev seed final bounds) (head : pid.z < p.B * p.H) :
    Safety.stepSafe (engine (withDot M D)) bounds (stateLoopStmt p scale rev seed final) s := by
  let invariant := fun (_ : Nat) (t : State α) =>
    StateReady pid input t ∧ ∃ values, HasCarry p rev values t
  have bodySafe : ∀ n, n < p.NT → ∀ t, invariant n t →
      Safety.traceSafeList (engine (withDot M D)) bounds
        (stateParts p scale rev seed final).body
        (t.setReg (loopName rev) .nat [] (fun _ => n)) := by
    intro n active t ⟨ready, values, carry⟩
    exact state_body_safe M D p scale rev seed final bounds pid input t ready values carry n
      active capacity head
  have preserved : ∀ n, n < p.NT → ∀ t, invariant n t → ∀ u,
      run (engine (withDot M D)) (stateParts p scale rev seed final).body
        (t.setReg (loopName rev) .nat [] (fun _ => n)) = some u → invariant (n + 1) u := by
    intro n _ t ⟨ready, values, carry⟩ u ran
    obtain ⟨actual, executed, nextReady, nextCarry, _⟩ :=
      stateStep M D p scale rev seed final pid input t n values ready carry
    have same := Option.some.inj (executed.symm.trans ran)
    subst u
    exact ⟨nextReady, _, nextCarry⟩
  have safe := Safety.rangeSafe_of_invariant (engine (withDot M D)) bounds
    (loopName rev) p.NT 1 (stateParts p scale rev seed final).body invariant
    bodySafe preserved 0 s ⟨ctx, h, carry⟩
  cases rev <;> simpa [stateLoopStmt, Safety.stepSafe_unfold, Safety.safeOp_unfold,
    evalOp_unfold] using safe

/-- Complete actual forward/reverse state source, including optional loads,
every loop iteration and optional final store, for arbitrary backing memory. -/
theorem state_source_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (rev seed final : Bool) (bounds : RegionName → Nat) (s : State α)
    (capacity : StateCapacity p rev seed final bounds) (head : s.pids 2 < p.B * p.H) :
    TraceSafe (engine (withDot M D)) bounds (stateKernel p scale rev seed final) s := by
  let pid : PID := ⟨s.pids 0, s.pids 1, s.pids 2⟩
  let start := stateEntry M D p scale rev seed final s
  have ready := stateEntry_ready M D p scale rev seed final s
  obtain ⟨mid, loopRan, midReady, midCarry, _⟩ :=
    stateLoop M D p scale rev seed final pid s.mem start.val ready.1 ready.2
  have loopStep : step (engine (withDot M D)) (stateLoopStmt p scale rev seed final) start.val =
      some mid := by
    cases rev <;> simpa [stateLoopStmt, step, evalOp_unfold, loopName, Option.bind] using loopRan
  unfold TraceSafe
  rw [state_parts, List.append_assoc, Safety.traceSafeList_append]
  refine ⟨state_entry_safe M D p scale rev seed final bounds s capacity head, ?_⟩
  intro t ran
  have same : start.val = t := Option.some.inj (start.property.symm.trans ran)
  subst t
  apply Safety.traceSafeList_cons_of_step
    (state_loop_safe M D p scale rev seed final bounds pid s.mem start.val ready.1
      _ ready.2 capacity head) loopStep
  exact state_finish_safe M D p scale rev seed final bounds pid s.mem mid midReady capacity head

#axiomsClean state_body_safe
#axiomsClean state_loop_safe
#axiomsClean state_source_safe
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.StageSafety
