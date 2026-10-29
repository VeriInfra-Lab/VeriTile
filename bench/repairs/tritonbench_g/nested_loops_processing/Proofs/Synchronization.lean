import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.Schedule

/-! Local CTA barrier semantics, separate from any output-value theorem.
The revised source places an unconditional tl.debug_barrier after every store.
The native boundary requires a CTA barrier (for example SM80 bar.sync 0),
with all CTA threads participating; this file does not inspect generated code.
Completion here is the visibility/completion ordered by that barrier, not merely
an instruction issue timestamp. No claim follows from one CTA/warp alone. -/
namespace VeriTile.Bench.Repairs.NestedLoopsProcessing
noncomputable section

/-- The three clauses are local to one barrier and the next source statement.
They are ordinary compiler/runtime synchronization laws. There is no output
value, target, final memory equation, or all-pairs store-order premise. -/
structure CtaBarrierContract where
  threads : Nat
  positive : 0 < threads
  writer : Lane → Fin threads
  completed : Nat → Lane → Nat
  arrival : Nat → Fin threads → Nat
  release : Nat → Nat
  beforeArrival : ∀ t, t < 24 → ∀ lane, completed t lane < arrival t (writer lane)
  allArrived : ∀ t, t < 24 → ∀ thread, arrival t thread ≤ release t
  nextStatement : ∀ t, t+1 < 24 → ∀ lane, release t < completed (t+1) lane

theorem CtaBarrierContract.before_release (barrier : CtaBarrierContract)
    (t : Nat) (bound : t < 24) (lane : Lane) :
    barrier.completed t lane < barrier.release t :=
  lt_of_lt_of_le (barrier.beforeArrival t bound lane)
    (barrier.allArrived t bound (barrier.writer lane))

theorem CtaBarrierContract.release_step (barrier : CtaBarrierContract)
    (t : Nat) (bound : t+1 < 24) : barrier.release t < barrier.release (t+1) := by
  exact lt_trans (barrier.nextStatement t bound (0,0))
    (barrier.before_release (t+1) bound (0,0))

theorem CtaBarrierContract.release_le (barrier : CtaBarrierContract)
    (t u : Nat) (order : t ≤ u) (bound : u < 24) :
    barrier.release t ≤ barrier.release u := by
  induction u, order using Nat.le_induction with
  | base => exact le_rfl
  | succ u order ih =>
    exact (ih (by omega)).trans (Nat.le_of_lt (barrier.release_step u bound))

/-- Conflicting writes by different threads are ordered by the actual barrier
epochs. This is derived from adjacent local clauses, not assumed globally. -/
theorem CtaBarrierContract.cross_store_order (barrier : CtaBarrierContract)
    (t u : Nat) (order : t < u) (bound : u < 24) (a b : Lane) :
    barrier.completed t a < barrier.completed u b := by
  have previous : t ≤ u-1 := by omega
  have same : u-1+1=u := by omega
  have first := barrier.before_release t (by omega) a
  have middle := barrier.release_le t (u-1) previous (by omega)
  have last := barrier.nextStatement (u-1) (by omega) b
  rw [same] at last
  exact lt_trans (lt_of_lt_of_le first middle) last

/-- An inhabited local clock fixture for the finite 24-barrier source trace.
It is not a witness that a GPU invocation succeeds or that any output is right. -/
def CtaBarrierContract.canonical : CtaBarrierContract where
  threads := 128
  positive := by decide
  writer := fun lane => ⟨2*lane.1.val+lane.2.val, by have := lane.1.isLt; have := lane.2.isLt; omega⟩
  completed := fun t _ => 4*t
  arrival := fun t _ => 4*t+1
  release := fun t => 4*t+2
  beforeArrival := by intros; omega
  allArrived := by intros; omega
  nextStatement := by intros; omega

/-- The independently reported r4/c4 conflict is separated even though its
earlier writer (1,0) and later writer (0,0) are different native threads. -/
theorem reported_conflict_order (barrier : CtaBarrierContract) :
    barrier.completed 1 (1,0) < barrier.completed 3 (0,0) :=
  barrier.cross_store_order 1 3 (by decide) (by decide) (1,0) (0,0)

#axiomsClean CtaBarrierContract.before_release
#axiomsClean CtaBarrierContract.release_step
#axiomsClean CtaBarrierContract.release_le
#axiomsClean CtaBarrierContract.cross_store_order
#axiomsClean CtaBarrierContract.canonical
#axiomsClean reported_conflict_order
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing
