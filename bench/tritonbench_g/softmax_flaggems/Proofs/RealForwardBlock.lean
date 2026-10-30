import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardTiles
import bench.tritonbench_g.softmax_flaggems.Proofs.FPTraceGrid

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.RealForward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
open FPTrace TraceReadback
noncomputable section
set_option maxHeartbeats 6000000

theorem block_sound (X Y : RegionName) (q : Nat → WithBot ℝ) (separate : X≠Y)
    (d : Dimensions) (c : LaunchConfig) (defined : Defined q d)
    (hN : 0<d.N) (hK : 0<d.K) (hB : 0<c.B) (hone : c.one=oneTile d.N c.B)
    (row col : Nat) (hr : row<d.M) (s : State (WithBot ℝ)) (input : Input X q s) :
    ∀ tile, tile∈forwardTiles Ideal.algebra X Y d c row col s → Sound (Input X q) (cells q d.N d.K) tile := by
  intro tile member
  unfold forwardTiles at member
  split at member <;> cases hc : c.one
  · simp only [hc,Bool.false_eq_true,if_false] at member
    have walk := noninner_walk s X row col d.N d.K c.B c.C hB
    simp only [noninner_seed] at walk
    rw [walk] at member
    obtain ⟨cur,hcur,rfl⟩ := List.mem_map.mp member
    exact online_noninner_tile s X Y q input separate row col d.N d.K c.B c.C cur hB
  · simp only [hc,if_true,List.mem_singleton] at member
    subst tile
    have covered : d.N≤c.B := by simp [hc,oneTile] at hone; exact hone
    exact one_noninner_tile X Y q separate row col d.N d.K c.B c.C hB covered (defined row hr)
  · have k1 : d.K=1 := by omega
    simp only [hc,Bool.false_eq_true,if_false] at member
    have walk := inner_walk s X row d.N c.B hN hB
    simp only [inner_seed] at walk
    rw [walk] at member
    rw [k1]
    rcases List.mem_append.mp member with before|after
    · obtain ⟨cur,hcur,rfl⟩ := List.mem_map.mp before
      apply online_inner_tile s X Y q input separate Bool.true row d.N c.B cur hB
      intro i hi
      simpa only [FPForwardInner.active,Bool.not_true,Bool.false_or,decide_eq_true_eq] using hi
    · obtain ⟨cur,hcur,rfl⟩ := List.mem_map.mp after
      apply online_inner_tile s X Y q input separate Bool.false row d.N c.B cur hB
      intro i hi
      have bounds := visits_bounds d.N c.B c.B cur hcur
      have prev : FPForwardInner.previous d.N c.B<d.N := RealAddress.previous_lt hN hB
      have lane := i.1.isLt
      omega
  · have k1 : d.K=1 := by omega
    simp only [hc,if_true,List.mem_singleton] at member
    subst tile
    have covered : d.N≤c.B := by simp [hc,oneTile] at hone; exact hone
    have hd := defined row hr 0 (by omega)
    rw [k1]
    apply one_inner_tile X Y q separate row d.N c.B hB covered
    change Target.Defined (fun n : Fin d.N => q (row*d.N*d.K+n.val*d.K+0)) at hd
    simpa only [k1,Nat.mul_one,Nat.add_zero] using hd

theorem grid_sound (X Y : RegionName) (q : Nat → WithBot ℝ) (separate : X≠Y)
    (d : Dimensions) (c : LaunchConfig) (defined : Defined q d)
    (hN : 0<d.N) (hK : 0<d.K) (hB : 0<c.B) (hone : c.one=oneTile d.N c.B)
    (ps : List Program) (rows : ∀ p, p∈ps → p.1<d.M) (s : State (WithBot ℝ)) (input : Input X q s) :
    ∀ tile, tile∈gridTiles (forwardTiles Ideal.algebra X Y d c) ps s →
      Sound (Input X q) (cells q d.N d.K) tile := by
  induction ps generalizing s with
  | nil => intro tile member; simp [gridTiles] at member
  | cons p ps ih =>
    have here := block_sound X Y q separate d c defined hN hK hB hone p.1 p.2 (rows p (List.mem_cons_self)) s input
    have next := (execute_reads (Input X q) (cells q d.N d.K) _ here s input).1
    intro tile member
    rcases List.mem_append.mp member with h|h
    · exact here tile h
    · exact ih (fun u hu => rows u (List.mem_cons_of_mem p hu)) _ next tile h

theorem whole_graph (X Y : RegionName) (q : Nat → WithBot ℝ) (separate : X≠Y)
    (d : Dimensions) (c : LaunchConfig) (defined : Defined q d)
    (hN : 0<d.N) (hK : 0<d.K) (hB : 0<c.B) (hC : 0<c.C) (hone : c.one=oneTile d.N c.B)
    (s : State (WithBot ℝ)) (input : Input X q s) :
    Input X q (FPGrid.forwardGraph Ideal.algebra X Y d c s) ∧
    ∀ r o, (FPGrid.forwardGraph Ideal.algebra X Y d c s).mem r o =
      if r=Y ∧ o<d.M*d.N*d.K then cells q d.N d.K r o else s.mem r o := by
  have rows : ∀ p, p∈grid (FPGrid.forwardGrid d c).1 (FPGrid.forwardGrid d c).2 → p.1<d.M := by
    intro p hp
    unfold FPGrid.forwardGrid at hp
    split at hp <;> exact (mem_grid.mp hp).1
  have sound := grid_sound X Y q separate d c defined hN hK hB hone _ rows s input
  obtain ⟨hi,readback⟩ := execute_reads (Input X q) (cells q d.N d.K) _ sound s input
  rw [FPTrace.forward_graph_trace]
  refine ⟨hi,?_⟩
  intro r o
  change ∀ r o, (execute (forwardTrace Ideal.algebra X Y d c s) s).mem r o =
    if (r,o)∈events (forwardTrace Ideal.algebra X Y d c s) then cells q d.N d.K r o else s.mem r o at readback
  rw [readback]
  simp only [show ((r,o)∈events (FPTrace.forwardTrace Ideal.algebra X Y d c s)) ↔ r=Y ∧ o<d.M*d.N*d.K from
    FPTrace.forward_events_exact Ideal.algebra X Y d c hN hK hB hC hone s (r,o)]

#axiomsClean block_sound
#axiomsClean grid_sound
#axiomsClean whole_graph
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.RealForward
