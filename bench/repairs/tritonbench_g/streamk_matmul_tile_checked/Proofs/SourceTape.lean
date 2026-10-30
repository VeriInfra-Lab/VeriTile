import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.SourceReads
import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.Proofs.ClosedRefinement

namespace VeriTile.Bench.Repairs.StreamK.TileChecked.ClosedSource
open Whole001 Source VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000

def firstReadTape (p : Parameters) : List Concurrent.Event :=
  if hs : 0 < (Plan.select p.h).streamTiles then
    let hI := Plan.iterations_positive p.h (Plan.selected_active p.h hs).2 p.blockK
    (List.finRange p.h.programs).flatMap fun program =>
      (Protocol.programChunks (Plan.select p.h).streamTiles (Plan.select p.h).iterations
        p.h.programs program.val hI).flatMap fun c =>
          tileReads p Reader.sourcePlan.mac.core (Reader.macEnv p Reader.sourcePlan.mac program.val c)
  else []

theorem firstReads_eq (p : Parameters) : firstReads p Reader.sourcePlan = some (firstReadTape p) := by
  by_cases active : 0 < (Plan.select p.h).streamTiles
  · have hI := Plan.iterations_positive p.h (Plan.selected_active p.h active).2 p.blockK
    unfold firstReads firstReadTape
    rw [if_pos active,dif_pos active]
    simp only [Reader.first_pid,Reader.first_stop,Reader.first_start,
      Reader.source_chunks p _ _ _ hI,Option.bind_eq_bind,Option.bind_some,Option.pure_def]
    rw [(Reader.selected_parts p).1,(Reader.selected_parts p).2]
    have mapPure (f : Fin p.h.programs → List Concurrent.Event) :
        (List.finRange p.h.programs).mapM (fun pid => some (f pid)) =
          some ((List.finRange p.h.programs).map f) :=
      List.mapM_pure (m:=Option) (f:=f)
    rw [mapPure,Option.bind_some]
    rfl
  · simp only [firstReads,firstReadTape,if_neg active,dif_neg active]

def completeTape (p : Parameters) : List Concurrent.Event :=
  if p.h.m=0 ∨ p.h.n=0 then [] else
    firstReadTape p ++ (firstWave p symbolic).map (lowerEvent p) ++
      (List.range (Plan.select p.h).streamTiles).flatMap (reduceReads p Reader.sourcePlan.reduce) ++
      (reduceWave p).map (lowerEvent p) ++ fullReads p Reader.sourcePlan.full ++
      (fullWave p symbolic).map (lowerEvent p)

theorem sourceTape_eq (p : Parameters) : sourceTape p = some (completeTape p) := by
  unfold sourceTape completeTape
  rw [Reader.source_compiles]
  simp only [Option.bind_eq_bind,Option.bind_some,Option.pure_def]
  split
  · rfl
  · rw [Reader.all_events]
    simp only [Option.bind_eq_bind,Option.bind_some,Option.pure_def,firstReads_eq]

/-- Only a classification of already decoded event data, not an execution or
access-safety assumption in the public contract. -/
def ReadOnly : Concurrent.Event → Prop
  | .readOnly _ => True
  | _ => False

theorem inputReads_readOnly (tile : Reader.Tile) (env : Reader.Env) (lane : Reader.Lanes)
    (iter : Nat) : ∀ e ∈ inputReads tile env lane iter, ReadOnly e := by
  intro e he
  unfold inputReads at he
  dsimp only at he
  split at he <;> split at he <;> simp_all [ReadOnly]
  rcases he with rfl|rfl <;> trivial

theorem flatMap_readOnly (xs : List ι) (f : ι → List Concurrent.Event)
    (h : ∀ i, ∀ e ∈ f i, ReadOnly e) : ∀ e ∈ xs.flatMap f, ReadOnly e := by
  intro e he
  obtain ⟨i,_,hi⟩ := List.mem_flatMap.mp he
  exact h i e hi

theorem tileReads_readOnly (p : Parameters) (tile : Reader.Tile) (env : Reader.Env) :
    ∀ e ∈ tileReads p tile env, ReadOnly e := by
  unfold tileReads
  apply flatMap_readOnly
  intro lane
  apply flatMap_readOnly
  intro iter
  apply flatMap_readOnly
  intro k
  exact inputReads_readOnly _ _ _ _

theorem firstReadTape_readOnly (p : Parameters) : ∀ e ∈ firstReadTape p, ReadOnly e := by
  unfold firstReadTape
  split
  · apply flatMap_readOnly
    intro program
    apply flatMap_readOnly
    intro c
    exact tileReads_readOnly _ _ _
  · simp

theorem reduceReads_readOnly (p : Parameters) (program : Reader.Reduce) (tile : Nat) :
    ∀ e ∈ reduceReads p program tile, ReadOnly e := by
  unfold reduceReads
  apply flatMap_readOnly
  intro pid
  apply flatMap_readOnly
  intro lane e he
  dsimp only at he
  split at he <;> simp_all [ReadOnly]

theorem fullReads_readOnly (p : Parameters) (program : Reader.Full) :
    ∀ e ∈ fullReads p program, ReadOnly e := by
  unfold fullReads
  apply flatMap_readOnly
  intro pid
  exact tileReads_readOnly _ _ _

theorem run_readOnly (backend : Concurrent.Backend α) (events : List Concurrent.Event)
    (h : ∀ e ∈ events, ReadOnly e) (s : Concurrent.State α) : Concurrent.run backend events s = s := by
  induction events generalizing s with
  | nil => rfl
  | cons e rest ih =>
    have he := h e (List.mem_cons_self ..)
    have hr : ∀ e ∈ rest,ReadOnly e := fun e he => h e (List.mem_cons_of_mem _ he)
    rw [Concurrent.run,ih hr]
    cases e <;> simp_all [ReadOnly,Concurrent.execute]

theorem run_append (backend : Concurrent.Backend α) (first second : List Concurrent.Event)
    (s : Concurrent.State α) :
    Concurrent.run backend (first++second) s = Concurrent.run backend second (Concurrent.run backend first s) := by
  induction first generalizing s with
  | nil => rfl
  | cons e rest ih => exact ih _

theorem complete_projection (p : Parameters) (ops : Instructions α) (exterior : Nat)
    (s : Concurrent.State α) :
    extract p exterior (Concurrent.run (backend ops) (completeTape p) s) = Source.call p ops (extract p exterior s) := by
  unfold completeTape Source.call
  split
  · rfl
  · simp only [run_append,run_readOnly _ _ (firstReadTape_readOnly p),
      run_readOnly _ _ (flatMap_readOnly _ _ (reduceReads_readOnly p Reader.sourcePlan.reduce)),
      run_readOnly _ _ (fullReads_readOnly p Reader.sourcePlan.full)]
    rw [run_projection p ops exterior _ _ (full_corresponds p ops),
      run_projection p ops exterior _ _ (reduce_corresponds p ops),
      run_projection p ops exterior _ _ (first_corresponds p ops)]

#axiomsClean firstReads_eq
#axiomsClean sourceTape_eq
#axiomsClean complete_projection
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.ClosedSource
