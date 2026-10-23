import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.FPTraceGrid

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.OutputTags
open VeriTile Triton FP.Structural
open SoftmaxFlaggemsFiniteSeed FPReference FPTrace
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section
variable {α ι : Type} [Inhabited α]

def RealCell (cell : Cell α) : Prop := ∃ v,cell=Cell.mk .real v

theorem write_keeps (s : State α) (Y r : RegionName) (offset k : Nat) (v : α)
    (h : RealCell (s.mem r k)) : RealCell ((s.write Y offset (.mk .real v)).mem r k) := by
  by_cases atCell : r=Y ∧ k=offset
  · exact ⟨v,by simp [State.write,atCell.1,atCell.2]⟩
  · simpa [State.write,atCell] using h

theorem fold_cells (xs : List ι) (Y r : RegionName) (offset : ι → Nat)
    (value : ι → α) (k : Nat) (s : State α)
    (h : (∃ i∈xs,r=Y ∧ k=offset i) ∨ RealCell (s.mem r k)) :
    RealCell ((xs.foldl (fun t i => t.write Y (offset i) (.mk .real (value i))) s).mem r k) := by
  induction xs generalizing s with
  | nil =>
    rcases h with h|h
    · simpa using h
    · exact h
  | cons i xs ih =>
    apply ih
    rcases h with ⟨j,hj,hr,hk⟩|initial
    · rcases List.mem_cons.mp hj with rfl|ht
      · right
        exact ⟨value j,by simp [State.write,hr,hk]⟩
      · exact Or.inl ⟨j,ht,hr,hk⟩
    · exact Or.inr (write_keeps s Y r (offset i) k (value i) initial)

theorem tile_cells (t : Tile α) (r : RegionName) (k : Nat) (s : State α)
    (h : (r,k)∈t.events ∨ RealCell (s.mem r k)) : RealCell ((t.execute s).mem r k) := by
  rw [t.exact_stores]
  apply fold_cells
  rcases h with event|initial
  · obtain ⟨i,active,same⟩ := (t.mem_events (r,k)).mp event
    left
    exact ⟨i,by simp [active,TileShape.mem_allIndices],congrArg Prod.fst same,congrArg Prod.snd same⟩
  · exact Or.inr initial

theorem trace_cells (ts : List (Tile α)) (r : RegionName) (k : Nat) (s : State α)
    (h : (r,k)∈events ts ∨ RealCell (s.mem r k)) : RealCell ((execute ts s).mem r k) := by
  induction ts generalizing s with
  | nil =>
    rcases h with event|initial
    · simpa [events] using event
    · exact initial
  | cons t ts ih =>
    apply ih
    rcases h with event|initial
    · have : (r,k)∈t.events ∨ (r,k)∈events ts := by simpa [events] using event
      rcases this with head|tail
      · exact Or.inr (tile_cells t r k s (Or.inl head))
      · exact Or.inl tail
    · exact Or.inr (tile_cells t r k s (Or.inr initial))

theorem forward (A : Algebra α) (X Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (hN : 0<d.N) (hK : 0<d.K) (hB : 0<c.B) (hC : 0<c.C) (hone : c.one=oneTile d.N c.B)
    (s : State α) (k : Nat) (hk : k<d.M*d.N*d.K) :
    RealCell ((FPGrid.forwardGraph A X Y d c s).mem Y k) := by
  rw [forward_graph_trace]
  apply trace_cells
  exact Or.inl ((forward_events_exact A X Y d c hN hK hB hC hone s (Y,k)).mpr ⟨rfl,hk⟩)

theorem backward (A : Algebra α) (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (hN : 0<d.N) (hK : 0<d.K) (hB : 0<c.B) (hC : 0<c.C) (hone : c.one=oneTile d.N c.B)
    (s : State α) (k : Nat) (hk : k<d.M*d.N*d.K) :
    RealCell ((FPGrid.backwardGraph A P G Y d c s).mem Y k) := by
  rw [backward_graph_trace]
  apply trace_cells
  exact Or.inl ((backward_events_exact A P G Y d c hN hK hB hC hone s (Y,k)).mpr ⟨rfl,hk⟩)

end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.OutputTags
