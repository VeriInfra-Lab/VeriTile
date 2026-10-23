import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.FPGraphMemory

namespace VeriTile.Bench.Repairs.SoftmaxFlaggems.FPTrace
open VeriTile Triton FP.Structural FPReference FPLoop
noncomputable section
set_option maxHeartbeats 3000000
variable {α β : Type} [Inhabited α]

structure Tile (α : Type) where
  shape : TileShape
  region : RegionName
  offset : TileIndex shape → Nat
  active : TileIndex shape → Bool
  value : State α → TileIndex shape → α

def Tile.events (t : Tile α) : List (RegionName × Nat) :=
  ((TileShape.allIndices t.shape).filter t.active).map (fun i => (t.region,t.offset i))

def Tile.execute (t : Tile α) (s : State α) : State α :=
  scatter s t.region t.offset t.active (t.value s)

/-- Every recorded event is an executed store, with all tile operands evaluated
before the first store. This is a derived equation, not a trace contract. -/
theorem Tile.exact_stores (t : Tile α) (s : State α) :
    t.execute s = ((TileShape.allIndices t.shape).filter t.active).foldl
      (fun st i => st.write t.region (t.offset i) (.mk .real (t.value s i))) s := by
  simpa [Tile.execute,scatter] using State.masked_scatter_eq_filter t.region t.offset
    (fun i => Cell.mk .real (t.value s i)) (fun i => t.active i=Bool.true)
    (TileShape.allIndices t.shape) s

theorem Tile.mem_events (t : Tile α) (a : RegionName × Nat) :
    a∈t.events ↔ ∃ i, t.active i=Bool.true ∧ a=(t.region,t.offset i) := by
  simp only [Tile.events,List.mem_map,List.mem_filter,TileShape.mem_allIndices,true_and]
  exact exists_congr (fun _ => and_congr_right (fun _ => eq_comm))

theorem Tile.frame (t : Tile α) (s : State α) (r : RegionName) (o : Nat)
    (h : (r,o)∉t.events) : (t.execute s).mem r o=s.mem r o := by
  unfold Tile.execute scatter
  apply State.masked_scatter_frame
  by_cases hr : r=t.region
  · right
    intro i _ hi he
    apply h
    rw [Tile.mem_events]
    exact ⟨i,hi,by simp [hr,he]⟩
  · exact Or.inl hr

def execute : List (Tile α) → State α → State α
  | [],s => s
  | t::ts,s => execute ts (t.execute s)

def events (ts : List (Tile α)) := ts.flatMap Tile.events

theorem frame (ts : List (Tile α)) (s : State α) (r : RegionName) (o : Nat)
    (h : (r,o)∉events ts) : (execute ts s).mem r o=s.mem r o := by
  induction ts generalizing s with
  | nil => rfl
  | cons t ts ih =>
    have hn : (r,o)∉t.events ∧ (r,o)∉events ts := by simpa [events] using h
    exact (ih (t.execute s) hn.2).trans (t.frame s r o hn.1)

def visits (stop stride cur : Nat) : List Nat :=
  if stride=0 then [] else if cur<stop then cur::visits stop stride (cur+stride) else []
termination_by stop-cur

theorem walk_tiles (tile : Nat → Tile α) (stop stride cur : Nat) (s : State α) :
    (walk (fun c st => (tile c).execute st) stop stride cur s).2 =
      execute ((visits stop stride cur).map tile) s := by
  by_cases hz : stride=0
  · simp [walk,visits,hz,execute]
  by_cases hc : cur<stop
  · conv => lhs; rw [walk,if_neg hz,if_pos hc]
    conv => rhs; rw [visits,if_neg hz,if_pos hc,List.map_cons,execute]
    exact walk_tiles tile stop stride (cur+stride) ((tile cur).execute s)
  · simp [walk,visits,hz,hc,execute]
termination_by stop-cur

theorem visits_contains (stop B cur k : Nat) (hB : 0<B) (hk : cur+k*B<stop) :
    cur+k*B ∈ visits stop B cur := by
  induction k generalizing cur with
  | zero => simp only [Nat.zero_mul,Nat.add_zero] at hk ⊢; rw [visits,if_neg (by omega),if_pos hk]; simp
  | succ k ih =>
    have hc : cur<stop := by omega
    rw [visits,if_neg (by omega),if_pos hc]
    apply List.mem_cons_of_mem
    have he : cur+(k+1)*B=(cur+B)+k*B := by simp [Nat.add_mul,Nat.add_assoc,Nat.add_left_comm,Nat.add_comm]
    rw [he] at hk ⊢
    exact ih (cur+B) hk

theorem visits_bounds (stop B cur x : Nat) (hx : x∈visits stop B cur) : cur≤x ∧ x<stop := by
  by_cases hz : B=0
  · simp [visits,hz] at hx
  by_cases hc : cur<stop
  · rw [visits,if_neg hz,if_pos hc] at hx
    rcases List.mem_cons.mp hx with h|h
    · subst x; exact ⟨le_rfl,hc⟩
    · have hb := visits_bounds stop B (cur+B) x h
      exact ⟨by omega,hb.2⟩
  · simp [visits,hz,hc] at hx
termination_by stop-cur

theorem execute_append (a b : List (Tile α)) (s : State α) :
    execute (a++b) s=execute b (execute a s) := by
  induction a generalizing s with
  | nil => rfl
  | cons t ts ih => exact ih (t.execute s)

#axiomsClean Tile.exact_stores
#axiomsClean frame
#axiomsClean visits_contains
#axiomsClean visits_bounds
end
end VeriTile.Bench.Repairs.SoftmaxFlaggems.FPTrace
