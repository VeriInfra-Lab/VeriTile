import bench.tritonbench_g.layer_norm_triton.Proofs.SourceFinalResult
import bench.tritonbench_g.layer_norm_triton.Proofs.SourceForwardReal
import bench.tritonbench_g.layer_norm_triton.Proofs.SourceRows
import bench.tritonbench_g.layer_norm_triton.Proofs.Affine

namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceFinalReal
open VeriTile.Triton FP.Structural SourceModel SourceFinalGraph
noncomputable section

structure Laws (m : Model ℝ) extends SourceForwardReal.Laws m where
  sumTile : ∀ BM BN (tile : Tile ℝ BM BN) (i : TileIndex [BN]),
    reduced m BM BN tile i=∑ j : Fin BM,tile (j,i)

def contribution (s : State ℝ) (r : RegionName) (M N BN start lane col : Nat) : ℝ :=
  if start+lane<M ∧ s.pids 0*BN+col<N then
    (s.mem r ((start+lane)*N+(s.pids 0*BN+col))).read .real else 0

theorem fold_values (m : Model ℝ) (laws : Laws m) (s : State ℝ) (M N BM BN cur : Nat)
    (a : Carries ℝ BM BN) (hb : 0<BM) (i : TileIndex [BM,BN]) :
    (fold m s M N BM BN cur a).1 i=a.1 i+
      ((SourceRows.rows M BM cur).map (fun start => contribution s "DW" M N BN start i.1.val i.2.1.val)).sum ∧
    (fold m s M N BM BN cur a).2 i=a.2 i+
      ((SourceRows.rows M BM cur).map (fun start => contribution s "DB" M N BN start i.1.val i.2.1.val)).sum := by
  rw [fold,SourceRows.rows]
  simp only [Nat.ne_of_gt hb,if_false]
  split
  · rename_i hc
    have ht := fold_values m laws s M N BM BN (cur+BM) (next m s M N BM BN cur a) hb i
    rw [ht.1,ht.2]
    simp only [next,loadTile,laws.add,laws.literal,List.map_cons,List.sum_cons,contribution]
    constructor <;> ring
  · rename_i hc
    simp [hc]
termination_by M-cur
decreasing_by omega

theorem lane_values (m : Model ℝ) (laws : Laws m) (s : State ℝ) (M N BN : Nat)
    (i : TileIndex [32,BN]) :
    (fold m s M N 32 BN 0 (zero m 32 BN)).1 i=
      ((groupRows M 32 0).map (fun start => contribution s "DW" M N BN start i.1.val i.2.1.val)).sum ∧
    (fold m s M N 32 BN 0 (zero m 32 BN)).2 i=
      ((groupRows M 32 0).map (fun start => contribution s "DB" M N BN start i.1.val i.2.1.val)).sum := by
  have h := fold_values m laws s M N 32 BN 0 (zero m 32 BN) (by decide) i
  simpa [zero,laws.literal,SourceRows.rows_eq M 32 0 (by decide)] using h

theorem range'_map (n : Nat) : List.range' 0 n 32=(List.range n).map (fun k => k*32) := by
  induction n with
  | zero => simp
  | succ n ih => simp [List.range'_concat,List.range_succ,ih,Nat.mul_comm]

theorem groupRows_map (M : Nat) : groupRows M 32 0=(List.range ((M+31)/32)).map (fun k => k*32) := by
  unfold groupRows
  have he : M-0+32-1=M+31 := by omega
  rw [he,range'_map]

theorem sum_contributions (s : State ℝ) (region : RegionName) (M N BN : Nat)
    (i : TileIndex [BN]) (hi : columns s BN i<N) :
    (∑ lane : Fin 32,((groupRows M 32 0).map
      (fun start => contribution s region M N BN start lane.val i.1.val)).sum)=
      ∑ row ∈ Finset.range M,(s.mem region (row*N+columns s BN i)).read .real := by
  simp only [groupRows_map,List.map_map,Function.comp_apply,RealProof.map_range_sum,contribution]
  have hcol : s.pids 0*BN+i.1.val<N := hi
  simp only [hcol,and_true]
  rw [Fin.sum_univ_eq_sum_range (fun lane => ∑ k ∈ Finset.range ((M+31)/32),
    if k*32+lane<M then (s.mem region ((k*32+lane)*N+(s.pids 0*BN+i.1.val))).read .real else 0)]
  rw [Finset.sum_comm,RealProof.tile_sum
    (fun j => if j<M then (s.mem region (j*N+(s.pids 0*BN+i.1.val))).read .real else 0)]
  apply VeriTile.Bench.TritonBenchG.LayerNormFwd.Formula.sum_padded
  have hc := Nat.div_add_mod (M+31) 32
  have hm := Nat.mod_lt (M+31) (by decide : 0<32)
  omega

theorem reduced_values (m : Model ℝ) (laws : Laws m) (s : State ℝ) (M N BN : Nat)
    (i : TileIndex [BN]) (hi : columns s BN i<N) :
    SourceForwardGraph.cast m (reduced m 32 BN (fold m s M N 32 BN 0 (zero m 32 BN)).1 i)=
      ∑ row ∈ Finset.range M,(s.mem "DW" (row*N+columns s BN i)).read .real ∧
    SourceForwardGraph.cast m (reduced m 32 BN (fold m s M N 32 BN 0 (zero m 32 BN)).2 i)=
      ∑ row ∈ Finset.range M,(s.mem "DB" (row*N+columns s BN i)).read .real := by
  simp only [laws.cast,laws.sumTile]
  constructor
  · simp only [(lane_values m laws s M N BN _).1]
    exact sum_contributions s "DW" M N BN i hi
  · simp only [(lane_values m laws s M N BN _).2]
    exact sum_contributions s "DB" M N BN i hi

#axiomsClean fold_values
#axiomsClean lane_values
#axiomsClean reduced_values
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceFinalReal
