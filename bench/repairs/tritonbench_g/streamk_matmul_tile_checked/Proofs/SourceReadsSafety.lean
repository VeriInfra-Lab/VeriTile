import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.Proofs.ClosedSafety

/-! Safety of every actual active quotation load. The read-only tape includes
all lanes, including lanes omitted by a masked output store or atomic. -/
namespace VeriTile.Bench.Repairs.StreamK.TileChecked.ClosedSource
open Whole001 Source VeriTile.Triton
noncomputable section
set_option maxHeartbeats 8000000

theorem inputReads_safe (p : Parameters) (tile : Reader.Tile) (env : Reader.Env)
    (lane : Reader.Lanes) (iter : Nat)
    (left : tile.loop.maskA.eval (Reader.loopEnv tile env lane iter) lane →
      tile.loop.left.index (Reader.loopEnv tile env lane iter) (Reader.loopPointers tile env lane iter) lane < p.h.m*p.h.k)
    (right : tile.loop.maskB.eval (Reader.loopEnv tile env lane iter) lane →
      tile.loop.right.index (Reader.loopEnv tile env lane iter) (Reader.loopPointers tile env lane iter) lane < p.h.k*p.h.n) :
    Concurrent.TapeSafe (bounds p) (inputReads tile env lane iter) := by
  unfold inputReads
  dsimp only
  apply tapeSafe_append
  · split
    next active =>
      intro e he
      simp only [List.mem_cons,List.not_mem_nil,or_false] at he
      subst e
      exact inWindow_nat _ _ (left active)
    next => simp [Concurrent.TapeSafe]
  · split
    next active =>
      intro e he
      simp only [List.mem_cons,List.not_mem_nil,or_false] at he
      subst e
      exact inWindow_nat _ _ (right active)
    next => simp [Concurrent.TapeSafe]

theorem mac_reads_safe (p : Parameters) (program : Nat) (c : Protocol.Chunk)
    (positive : 0 < (Plan.select p.h).iterations)
    (chunk : c.hi ≤ Protocol.boundary (Plan.select p.h).iterations c.lo) :
    Concurrent.TapeSafe (bounds p)
      (tileReads p Reader.sourcePlan.mac.core (Reader.macEnv p Reader.sourcePlan.mac program c)) := by
  unfold tileReads
  apply tapeSafe_flatMap
  intro lane _
  change Concurrent.TapeSafe _ ((List.range' c.lo (c.hi-c.lo)).flatMap _)
  apply tapeSafe_flatMap
  intro iter hi
  obtain ⟨offset,hoff,heq⟩ := List.mem_range'.mp hi
  simp only [Nat.one_mul] at heq
  have inside : c.lo ≤ iter ∧ iter < c.hi := by omega
  apply tapeSafe_flatMap
  intro k _
  exact inputReads_safe p _ _ _ _
    (Reader.mac_loadA_bound p program c lane k iter positive chunk inside)
    (Reader.mac_loadB_bound p program c lane k iter positive chunk inside)

theorem firstReads_safe (p : Parameters) : Concurrent.TapeSafe (bounds p) (firstReadTape p) := by
  unfold firstReadTape
  split
  next active =>
    apply tapeSafe_flatMap
    intro program _
    apply tapeSafe_flatMap
    intro c hc
    have positive := Plan.iterations_positive p.h (Plan.selected_active p.h active).2 p.blockK
    have shape := Protocol.member_shape _ _ _ positive c hc
    have bound : c.hi ≤ Protocol.boundary (Plan.select p.h).iterations c.lo := by
      rw [shape.2.2.2,Protocol.next_eq _ _ _ positive]
      exact min_le_left _ _
    exact mac_reads_safe p program.val c positive bound
  next => simp [Concurrent.TapeSafe]

theorem full_tileReads_safe (p : Parameters) (pid : Nat) :
    Concurrent.TapeSafe (bounds p)
      (tileReads p Reader.sourcePlan.full.core (Reader.fullEnv p Reader.sourcePlan.full pid)) := by
  unfold tileReads
  apply tapeSafe_flatMap
  intro lane _
  apply tapeSafe_flatMap
  intro iter _
  apply tapeSafe_flatMap
  intro k _
  exact inputReads_safe p _ _ _ _ (Reader.full_loadA_bound p pid lane k iter)
    (Reader.full_loadB_bound p pid lane k iter)

theorem fullReads_safe (p : Parameters) :
    Concurrent.TapeSafe (bounds p) (fullReads p Reader.sourcePlan.full) := by
  unfold fullReads
  exact tapeSafe_flatMap _ _ _ (fun pid _ => full_tileReads_safe p pid)

/-- The producer condition controls this load; the subsequent atomic's output
mask is deliberately absent. -/
theorem reduceReads_safe (p : Parameters) (tile : Nat) :
    Concurrent.TapeSafe (bounds p) (reduceReads p Reader.sourcePlan.reduce tile) := by
  unfold reduceReads
  change Concurrent.TapeSafe _ ((List.range' 0 (p.h.programs-0)).flatMap _)
  apply tapeSafe_flatMap
  intro pid hp
  obtain ⟨offset,hoff,heq⟩ := List.mem_range'.mp hp
  simp only [Nat.one_mul] at heq
  have hp : pid < p.h.programs := by omega
  apply tapeSafe_flatMap
  intro lane _
  dsimp only
  split
  · intro e he
    simp only [List.mem_cons,List.not_mem_nil,or_false] at he
    subst e
    exact inWindow_nat _ _ (Reader.reduce_load_bound p tile ⟨pid,hp⟩ lane)
  · simp [Concurrent.TapeSafe]

theorem complete_safe (p : Parameters) : Concurrent.TapeSafe (bounds p) (completeTape p) := by
  unfold completeTape
  split
  · simp [Concurrent.TapeSafe]
  · apply tapeSafe_append
    · apply tapeSafe_append
      · apply tapeSafe_append
        · apply tapeSafe_append
          · exact tapeSafe_append _ _ _ (firstReads_safe p) (first_safe p)
          · exact tapeSafe_flatMap _ _ _ (fun tile _ => reduceReads_safe p tile)
        · exact reduce_safe p
      · exact fullReads_safe p
    · exact full_safe p

#axiomsClean mac_reads_safe
#axiomsClean firstReads_safe
#axiomsClean reduceReads_safe
#axiomsClean fullReads_safe
#axiomsClean complete_safe
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.ClosedSource
