import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.RealBackwardTiles

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.RealBackward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
open FPTrace TraceReadback
noncomputable section
set_option maxHeartbeats 6000000

theorem block_sound (P G Y : RegionName) (p g : Nat → ℝ) (hPY : P≠Y) (hGY : G≠Y)
    (d : Dimensions) (c : LaunchConfig) (hK : 0<d.K) (hB : 0<c.B) (hone : c.one=oneTile d.N c.B)
    (row col : Nat) (hr : 1<d.K → row<d.M) (s : State (WithBot ℝ)) (input : Input P G p g (d.M*d.N*d.K) s) :
    ∀ tile, tile∈backwardTiles Ideal.algebra P G Y d c row col s → Sound (Input P G p g (d.M*d.N*d.K)) (cells p g d.N d.K) tile := by
  intro tile member
  unfold backwardTiles at member
  split at member <;> cases hc : c.one
  · rename_i noninner
    simp only [hc,Bool.false_eq_true,if_false] at member
    rw [noninner_walk s P G p g row col d.M d.N d.K c.B c.C hB (hr noninner) input] at member
    obtain ⟨cur,hcur,rfl⟩ := List.mem_map.mp member
    exact online_noninner_tile P G Y p g hPY hGY row col d.M d.N d.K c.B c.C cur hB (hr noninner)
  · rename_i noninner
    simp only [hc,if_true,List.mem_singleton] at member
    subst tile
    have covered : d.N≤c.B := by simp [hc,oneTile] at hone; exact hone
    exact one_noninner_tile P G Y p g hPY hGY row col d.M d.N d.K c.B c.C hB covered (hr noninner)
  · have k1 : d.K=1 := by omega
    simp only [hc,Bool.false_eq_true,if_false] at member
    have hi : Input P G p g (d.M*d.N) s := by simpa only [k1,Nat.mul_one] using input
    rw [inner_walk s P G p g row d.M d.N c.B c.C hB hi] at member
    obtain ⟨cur,hcur,rfl⟩ := List.mem_map.mp member
    simpa only [k1,Nat.mul_one] using online_inner_tile P G Y p g hPY hGY row d.M d.N c.B c.C cur hB
  · have k1 : d.K=1 := by omega
    simp only [hc,if_true,List.mem_singleton] at member
    subst tile
    have covered : d.N≤c.B := by simp [hc,oneTile] at hone; exact hone
    simpa only [k1,Nat.mul_one] using one_inner_tile P G Y p g hPY hGY row d.M d.N c.B c.C hB covered

theorem grid_sound (P G Y : RegionName) (p g : Nat → ℝ) (hPY : P≠Y) (hGY : G≠Y)
    (d : Dimensions) (c : LaunchConfig) (hK : 0<d.K) (hB : 0<c.B) (hone : c.one=oneTile d.N c.B)
    (ps : List Program) (rows : ∀ u, u∈ps → 1<d.K → u.1<d.M) (s : State (WithBot ℝ))
    (input : Input P G p g (d.M*d.N*d.K) s) :
    ∀ tile, tile∈gridTiles (backwardTiles Ideal.algebra P G Y d c) ps s →
      Sound (Input P G p g (d.M*d.N*d.K)) (cells p g d.N d.K) tile := by
  induction ps generalizing s with
  | nil => intro tile member; simp [gridTiles] at member
  | cons u ps ih =>
    have here := block_sound P G Y p g hPY hGY d c hK hB hone u.1 u.2 (rows u (List.mem_cons_self)) s input
    have next := (execute_reads (Input P G p g (d.M*d.N*d.K)) (cells p g d.N d.K) _ here s input).1
    intro tile member
    rcases List.mem_append.mp member with h|h
    · exact here tile h
    · exact ih (fun v hv => rows v (List.mem_cons_of_mem u hv)) _ next tile h

theorem whole_graph (P G Y : RegionName) (p g : Nat → ℝ) (hPY : P≠Y) (hGY : G≠Y)
    (d : Dimensions) (c : LaunchConfig) (hN : 0<d.N) (hK : 0<d.K) (hB : 0<c.B) (hC : 0<c.C)
    (hone : c.one=oneTile d.N c.B) (s : State (WithBot ℝ)) (input : Input P G p g (d.M*d.N*d.K) s) :
    Input P G p g (d.M*d.N*d.K) (FPGrid.backwardGraph Ideal.algebra P G Y d c s) ∧
    ∀ r o, (FPGrid.backwardGraph Ideal.algebra P G Y d c s).mem r o =
      if r=Y ∧ o<d.M*d.N*d.K then cells p g d.N d.K r o else s.mem r o := by
  have rows : ∀ u, u∈grid (FPGrid.backwardGrid d c).1 (FPGrid.backwardGrid d c).2 → 1<d.K → u.1<d.M := by
    intro u hu hk
    have bounds : u.1<d.M ∧ u.2<RealAddress.rounds d.K c.C := by
      apply mem_grid.mp
      simpa [FPGrid.backwardGrid,hk] using hu
    exact bounds.1
  have sound := grid_sound P G Y p g hPY hGY d c hK hB hone _ rows s input
  obtain ⟨hi,readback⟩ := execute_reads (Input P G p g (d.M*d.N*d.K)) (cells p g d.N d.K) _ sound s input
  rw [FPTrace.backward_graph_trace]
  refine ⟨hi,?_⟩
  intro r o
  change ∀ r o, (execute (backwardTrace Ideal.algebra P G Y d c s) s).mem r o =
    if (r,o)∈events (backwardTrace Ideal.algebra P G Y d c s) then cells p g d.N d.K r o else s.mem r o at readback
  rw [readback]
  simp only [show ((r,o)∈events (backwardTrace Ideal.algebra P G Y d c s)) ↔ r=Y ∧ o<d.M*d.N*d.K from
    FPTrace.backward_events_exact Ideal.algebra P G Y d c hN hK hB hC hone s (r,o)]

#axiomsClean block_sound
#axiomsClean grid_sound
#axiomsClean whole_graph
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.RealBackward
