import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.FPTraceCoverage
import bench.tritonbench_g.softmax_flaggems.Proofs.HostGridDispatch

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPTrace
open VeriTile Triton FP.Structural FPReference
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

/-- Flatten the actual per-CTA tile writes, threading the same memory snapshot
used by the independent value graph. Each Tile has exact store semantics. -/
def gridTiles (tiles : Nat → Nat → State α → List (Tile α)) : List Program → State α → List (Tile α)
  | [],_ => []
  | p::ps,s => let ts := tiles p.1 p.2 s
              ts ++ gridTiles tiles ps (execute ts s)

theorem grid_eq (block : Nat → Nat → State α → State α)
    (tiles : Nat → Nat → State α → List (Tile α))
    (h : ∀ r k s, block r k s=execute (tiles r k s) s) (ps : List Program) (s : State α) :
    FPGrid.graphRun block ps s=execute (gridTiles tiles ps s) s := by
  induction ps generalizing s with
  | nil => rfl
  | cons p ps ih =>
    simp only [FPGrid.graphRun,gridTiles,execute_append,h,ih]

theorem grid_mem_bounds (tiles : Nat → Nat → State α → List (Tile α))
    (ps : List Program) (allowed : RegionName × Nat → Prop)
    (h : ∀ p∈ps, ∀ s a, a∈events (tiles p.1 p.2 s) → allowed a)
    (s : State α) (a : RegionName × Nat) (ha : a∈events (gridTiles tiles ps s)) : allowed a := by
  induction ps generalizing s with
  | nil => simp [gridTiles,events] at ha
  | cons p ps ih =>
    rcases (mem_events_append _ _ _).mp ha with hp|ht
    · exact h p (List.mem_cons_self) s a hp
    · exact ih (fun q hq => h q (List.mem_cons_of_mem p hq)) _ ht

theorem grid_mem_contains (tiles : Nat → Nat → State α → List (Tile α))
    (ps : List Program) (p : Program) (hp : p∈ps) (a : RegionName × Nat)
    (h : ∀ s, a∈events (tiles p.1 p.2 s)) (s : State α) :
    a∈events (gridTiles tiles ps s) := by
  induction ps generalizing s with
  | nil => simp at hp
  | cons q ps ih =>
    rw [gridTiles,mem_events_append]
    rcases List.mem_cons.mp hp with he|ht
    · subst p; exact Or.inl (h s)
    · exact Or.inr (ih ht _)

def forwardTrace (A : Algebra α) (X Y : RegionName) (d : Dimensions) (c : LaunchConfig) (s : State α) :=
  let g := FPGrid.forwardGrid d c
  gridTiles (forwardTiles A X Y d c) (grid g.1 g.2) s

def backwardTrace (A : Algebra α) (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig) (s : State α) :=
  let g := FPGrid.backwardGrid d c
  gridTiles (backwardTiles A P G Y d c) (grid g.1 g.2) s

theorem forward_graph_trace (A : Algebra α) (X Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (s : State α) : FPGrid.forwardGraph A X Y d c s=execute (forwardTrace A X Y d c s) s :=
  grid_eq _ _ (forward_eq A X Y d c) _ s

theorem backward_graph_trace (A : Algebra α) (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (s : State α) : FPGrid.backwardGraph A P G Y d c s=execute (backwardTrace A P G Y d c s) s :=
  grid_eq _ _ (backward_eq A P G Y d c) _ s

theorem forward_events_exact (A : Algebra α) (X Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (hN : 0<d.N) (hK : 0<d.K) (hB : 0<c.B) (hC : 0<c.C) (hone : c.one=oneTile d.N c.B)
    (s : State α) (a : RegionName × Nat) :
    a∈events (forwardTrace A X Y d c s) ↔ a.1=Y ∧ a.2<d.M*d.N*d.K := by
  constructor
  · apply grid_mem_bounds
    intro p hp base a ha
    have hr : p.1<d.M := by
      unfold FPGrid.forwardGrid at hp
      split at hp <;> exact (mem_grid.mp hp).1
    exact forward_bounds A X Y d c hN hK hB hone p.1 p.2 hr base a ha
  · rintro ⟨hr,ho⟩
    obtain ⟨hm,hn,hk,he⟩ := coordinate_noninner hN hK ho
    let p := forwardOwner d c (a.2/d.K/d.N) (a.2%d.K)
    have hp := forward_owner_mem d c hC (a.2/d.K/d.N) (a.2%d.K) hm hk
    apply grid_mem_contains _ _ p hp a
    intro base
    have hc := forward_contains A X Y d c hB hC hone (a.2/d.K/d.N) (a.2/d.K%d.N)
      (a.2%d.K) hn hk base
    simpa only [he,←hr,Prod.mk.eta] using hc

theorem backward_events_exact (A : Algebra α) (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (hN : 0<d.N) (hK : 0<d.K) (hB : 0<c.B) (hC : 0<c.C) (hone : c.one=oneTile d.N c.B)
    (s : State α) (a : RegionName × Nat) :
    a∈events (backwardTrace A P G Y d c s) ↔ a.1=Y ∧ a.2<d.M*d.N*d.K := by
  constructor
  · apply grid_mem_bounds
    intro p hp base a ha
    have hr : 1<d.K → p.1<d.M := by
      intro h
      have hh : p.1<d.M ∧ p.2<RealAddress.rounds d.K c.C := by
        apply mem_grid.mp
        simpa [FPGrid.backwardGrid,h] using hp
      exact hh.1
    exact backward_bounds A P G Y d c hK p.1 p.2 hr base a ha
  · rintro ⟨hr,ho⟩
    obtain ⟨hm,hn,hk,he⟩ := coordinate_noninner hN hK ho
    let p := backwardOwner d c (a.2/d.K/d.N) (a.2%d.K)
    have hp := backward_owner_mem d c hC (a.2/d.K/d.N) (a.2%d.K) hm hk
    apply grid_mem_contains _ _ p hp a
    intro base
    have hc := backward_contains A P G Y d c hB hC hone (a.2/d.K/d.N) (a.2/d.K%d.N)
      (a.2%d.K) hm hn hk base
    simpa only [he,←hr,Prod.mk.eta] using hc

def Frame (Y : RegionName) (count : Nat) (s t : State α) : Prop :=
  ∀ r o, ¬(r=Y ∧ o<count) → t.mem r o=s.mem r o

theorem forward_frame (A : Algebra α) (X Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (hN : 0<d.N) (hK : 0<d.K) (hB : 0<c.B) (hC : 0<c.C) (hone : c.one=oneTile d.N c.B)
    (s : State α) : Frame Y (d.M*d.N*d.K) s (FPGrid.forwardGraph A X Y d c s) := by
  intro r o hn
  rw [forward_graph_trace]
  exact frame _ s r o (by rw [forward_events_exact A X Y d c hN hK hB hC hone]; exact hn)

theorem backward_frame (A : Algebra α) (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (hN : 0<d.N) (hK : 0<d.K) (hB : 0<c.B) (hC : 0<c.C) (hone : c.one=oneTile d.N c.B)
    (s : State α) : Frame Y (d.M*d.N*d.K) s (FPGrid.backwardGraph A P G Y d c s) := by
  intro r o hn
  rw [backward_graph_trace]
  exact frame _ s r o (by rw [backward_events_exact A P G Y d c hN hK hB hC hone]; exact hn)

#axiomsClean forward_graph_trace
#axiomsClean backward_graph_trace
#axiomsClean forward_events_exact
#axiomsClean backward_events_exact
#axiomsClean forward_frame
#axiomsClean backward_frame
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPTrace
