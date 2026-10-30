import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.Proofs.SourceTape
import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.Proofs.SourceAccess

namespace VeriTile.Bench.Repairs.StreamK.TileChecked.ClosedSource
open Whole001 Source VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000

/-- Immutable source arrays and logical output/scratch capacities. Physical
byte-window transport remains the existing ValueMemory/ByteExecution proof. -/
def bounds (p : Parameters) : Concurrent.Bounds where
  input := fun r => if r=0 then p.h.m*p.h.k else if r=1 then p.h.k*p.h.n else 0
  mem := fun r => if r=0 then p.h.m*p.h.n else if r=1 then p.h.programs*p.h.bm*p.h.bn else 0
  words := fun _ => 0

theorem inWindow_nat (n i : Nat) (h : i<n) : Concurrent.InWindow n i := by
  exact ⟨Int.natCast_nonneg i,by exact_mod_cast h⟩

theorem safe_loadA (p : Parameters) (row k : Nat) :
    (loadA p symbolic.zero (inputA p) row k).Safe (bounds p) := by
  unfold loadA
  split
  next active => exact inWindow_nat _ _ (Geometry.input_mask_safe _ _ _ _ _ _ p.inputA active.1 active.2)
  next => trivial

theorem safe_loadB (p : Parameters) (k col : Nat) :
    (loadB p symbolic.zero (inputB p) k col).Safe (bounds p) := by
  unfold loadB
  split
  next active => exact inWindow_nat _ _ (Geometry.input_mask_safe _ _ _ _ _ _ p.inputB active.1 active.2)
  next => trivial

theorem fold_safe (xs : List ι) (f : Concurrent.Expr → ι → Concurrent.Expr)
    (step : ∀ value i,value.Safe bs → (f value i).Safe bs)
    (seed : Concurrent.Expr) (initial : seed.Safe bs) : (xs.foldl f seed).Safe bs := by
  induction xs generalizing seed with
  | nil => exact initial
  | cons i rest ih => exact ih _ (step seed i initial)

theorem block_safe (p : Parameters) (tile iter : Nat) (lane : Lane p)
    (seed : Concurrent.Expr) (initial : seed.Safe (bounds p)) :
    (block p symbolic (inputA p) (inputB p) tile iter lane seed).Safe (bounds p) := by
  unfold block
  apply fold_safe
  · intro value group hv
    exact ⟨fun _ => safe_loadA p _ _,fun _ => safe_loadB p _ _,hv⟩
  · exact initial

theorem chunk_safe (p : Parameters) (tile lo hi : Nat) (lane : Lane p) :
    (chunkValue p symbolic (inputA p) (inputB p) tile lo hi lane).Safe (bounds p) := by
  unfold chunkValue
  apply fold_safe
  · intro value iter hv
    exact block_safe p tile iter lane value hv
  · trivial

theorem tapeSafe_append (bs : Concurrent.Bounds) (a b : List Concurrent.Event)
    (ha : Concurrent.TapeSafe bs a) (hb : Concurrent.TapeSafe bs b) : Concurrent.TapeSafe bs (a++b) := by
  intro e he
  rcases List.mem_append.mp he with he|he
  · exact ha e he
  · exact hb e he

theorem tapeSafe_flatMap (bs : Concurrent.Bounds) (xs : List ι) (f : ι → List Concurrent.Event)
    (h : ∀ i ∈ xs,Concurrent.TapeSafe bs (f i)) : Concurrent.TapeSafe bs (xs.flatMap f) := by
  intro e he
  obtain ⟨i,hi,he⟩ := List.mem_flatMap.mp he
  exact h i hi e he

theorem storeTile_safe (p : Parameters) (tile : Nat)
    (value : Lane p → (A p → Concurrent.Expr) → (B p → Concurrent.Expr) → Concurrent.Expr)
    (safe : ∀ lane,(value lane (inputA p) (inputB p)).Safe (bounds p)) :
    Concurrent.TapeSafe (bounds p) ((storeTile p value tile).map (lowerEvent p)) := by
  intro e he
  obtain ⟨event,hm,rfl⟩ := List.mem_map.mp he
  obtain ⟨lane,_,hl⟩ := List.mem_flatMap.mp hm
  split at hl
  · simp only [List.mem_cons,List.not_mem_nil,or_false] at hl
    subst event
    exact ⟨inWindow_nat _ _ (output p tile lane ‹_›).isLt,safe lane⟩
  · simp at hl

theorem mac_safe (p : Parameters) (program : Fin p.h.programs) (c : Protocol.Chunk) :
    Concurrent.TapeSafe (bounds p) ((macLoop p symbolic program c).map (lowerEvent p)) := by
  unfold macLoop
  dsimp only
  split
  · exact storeTile_safe p _ _ (fun lane => chunk_safe p _ _ _ lane)
  · intro e he
    obtain ⟨event,hm,rfl⟩ := List.mem_map.mp he
    obtain ⟨lane,_,rfl⟩ := List.mem_map.mp hm
    exact ⟨inWindow_nat _ _ (scratch p program lane).isLt,chunk_safe p _ _ _ lane⟩

theorem first_safe (p : Parameters) :
    Concurrent.TapeSafe (bounds p) ((firstWave p symbolic).map (lowerEvent p)) := by
  unfold firstWave
  split
  · simp only [List.map_flatMap]
    apply tapeSafe_flatMap
    intro program _
    apply tapeSafe_flatMap
    intro c _
    exact mac_safe p program c
  · simp [Concurrent.TapeSafe]

theorem reduce_safe (p : Parameters) :
    Concurrent.TapeSafe (bounds p) ((reduceWave p).map (lowerEvent p)) := by
  intro e he
  obtain ⟨event,hm,rfl⟩ := List.mem_map.mp he
  obtain ⟨pair,_,rfl⟩ := List.mem_map.mp hm
  exact ⟨inWindow_nat _ _ pair.1.isLt,inWindow_nat _ _ pair.2.isLt⟩

theorem full_safe (p : Parameters) :
    Concurrent.TapeSafe (bounds p) ((fullWave p symbolic).map (lowerEvent p)) := by
  unfold fullWave
  simp only [List.map_flatMap]
  apply tapeSafe_flatMap
  intro pid _
  exact storeTile_safe p _ _ (fun lane => chunk_safe p _ _ _ lane)

#axiomsClean chunk_safe
#axiomsClean first_safe
#axiomsClean reduce_safe
#axiomsClean full_safe
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.ClosedSource
