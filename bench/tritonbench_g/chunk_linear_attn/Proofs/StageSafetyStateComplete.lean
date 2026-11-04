import bench.tritonbench_g.chunk_linear_attn.Proofs.StageSafetyState
import bench.tritonbench_g.chunk_linear_attn.Proofs.SignedRangeSafety

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
      (stateIndex p rev n s) := by
  have split : (stateParts p scale rev seed final).body =
      stateStorePrefix p scale rev seed final ++ stateUpdateSuffix p scale rev seed final :=
    (List.take_append_drop _ _).symm
  rw [split, Safety.traceSafeList_append]
  refine ⟨state_store_safe M D p scale rev seed final bounds pid input s ctx n active capacity head, ?_⟩
  intro t ran
  let start := stateStoreExecution M D p scale rev seed final pid input s n h ctx carry
  have same : start.val = t := Option.some.inj (start.property.symm.trans ran)
  subst t
  have ready := stateStore_ready M D p scale rev seed final pid input s n h ctx carry active
  have pointers := stateStore_pointers M D p scale rev seed final pid input s n h ctx carry active
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
        (stateIndex p rev n t) := by
    intro n active t ⟨ready, values, carry⟩
    exact state_body_safe M D p scale rev seed final bounds pid input t ready values carry n
      active capacity head
  have preserved : ∀ n, n < p.NT → ∀ t, invariant n t → ∀ u,
      run (engine (withDot M D)) (stateParts p scale rev seed final).body
        (stateIndex p rev n t) = some u → invariant (n + 1) u := by
    intro n active t ⟨ready, values, carry⟩ u ran
    obtain ⟨actual, executed, nextReady, nextCarry, _⟩ :=
      stateStep M D p scale rev seed final pid input t n values ready carry active
    have same := Option.some.inj (executed.symm.trans ran)
    subst u
    exact ⟨nextReady, _, nextCarry⟩
  cases rev with
  | «false» =>
    have safe := Safety.rangeSafe_of_invariant (engine (withDot M D)) bounds
      "i_t" p.NT 1 (stateParts p scale Bool.false seed final).body invariant
      (by intro n hn t ht; simpa only [stateIndex,Bool.false_eq_true,if_false] using bodySafe n hn t ht)
      (by intro n hn t ht u ran; exact preserved n hn t ht u ran)
      0 s ⟨ctx, h, carry⟩
    simpa [stateLoopStmt, Safety.stepSafe_unfold, Safety.safeOp_unfold,
      evalOp_unfold] using safe
  | «true» =>
    have safe := SignedRange.clock_safe (engine (withDot M D)) bounds "__rev_i_t"
      (stateParts p scale Bool.true seed final).body (1-(p.NT : Int)) p.NT invariant
      (by
        intro n hn t ht
        simpa [stateIndex,add_comm,add_left_comm,add_assoc,sub_eq_add_neg] using bodySafe n hn t ht)
      (by
        intro n hn t ht u ran
        apply preserved n hn t ht u
        simpa [stateIndex,add_comm,add_left_comm,add_assoc,sub_eq_add_neg] using ran)
      0 (Nat.zero_le _) s ⟨ctx,h,carry⟩
    have first : (stateIntParts p scale seed final).start =
        .sub .int .nil (.constInt 0) (.sub .int .nil (.constInt (Int.ofNat p.NT)) (.constInt 1)) := rfl
    have last : (stateIntParts p scale seed final).stop = .constInt 1 := rfl
    have stride : (stateIntParts p scale seed final).stride = .constNat 1 := rfl
    simp only [stateLoopStmt,↓reduceIte,Safety.stepSafe_unfold,first,last,stride,
      Safety.safeOp_unfold,evalOp_unfold,numeric,bop,Option.bind,Option.bind_some,
      true_and,and_true]
    have beginEq : (0 : Int)-((p.NT : Int)-1)=(1-(p.NT : Int))+(0 : Nat) := by omega
    have endEq : (1-(p.NT : Int))+(p.NT : Int)=1 := by omega
    change Safety.rangeIntSafe (engine (withDot M D)) bounds "__rev_i_t"
      (0-((p.NT : Int)-1)) 1 1 (stateIntParts p scale seed final).body s
    rw [beginEq]
    simpa only [endEq] using safe

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
  unfold TraceSafe
  rw [state_parts, List.append_assoc, Safety.traceSafeList_append]
  refine ⟨state_entry_safe M D p scale rev seed final bounds s capacity head, ?_⟩
  intro t ran
  have same : start.val = t := Option.some.inj (start.property.symm.trans ran)
  subst t
  apply Safety.traceSafeList_cons_of_step
    (state_loop_safe M D p scale rev seed final bounds pid s.mem start.val ready.1
      _ ready.2 capacity head) loopRan
  exact state_finish_safe M D p scale rev seed final bounds pid s.mem mid midReady capacity head

#axiomsClean state_body_safe
#axiomsClean state_loop_safe
#axiomsClean state_source_safe
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.StageSafety
