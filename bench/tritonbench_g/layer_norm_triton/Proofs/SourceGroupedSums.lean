import bench.tritonbench_g.layer_norm_triton.Proofs.SourceGroupedReal
import bench.tritonbench_g.layer_norm_triton.Proofs.SourceRows

namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceGroupedSums
open VeriTile.Triton FP.Structural SourceModel SourceGroupedGraph SourceGroupedExecution SourceGroupedResult SourceGroupedReal
noncomputable section

theorem row_after_store (m : Model ℝ) (s : State ℝ) (stride N B cur r : Nat)
    (region : RegionName) (hr : region≠"DX") :
    SourceGroupedReal.row (storeDX m s stride N B cur) region stride r=SourceGroupedReal.row s region stride r := by
  funext j
  exact congrArg (fun cell : Cell ℝ => cell.read .real)
    (storeDX_frame m s stride N B cur region (r*stride+j) (Or.inl hr))

theorem normalized_after_store (m : Model ℝ) (s : State ℝ) (stride N B cur r j : Nat) :
    normalized (storeDX m s stride N B cur) stride r j=normalized s stride r j := by
  unfold normalized
  rw [row_after_store m s stride N B cur r "X" (by decide)]
  simp only [savedMean,savedInverse]
  rw [storeDX_frame m s stride N B cur "Mean" r (Or.inl (by decide)),
    storeDX_frame m s stride N B cur "Rstd" r (Or.inl (by decide))]

def weightTerm (s : State ℝ) (stride j : Nat) (r : Nat) : ℝ :=
  SourceGroupedReal.row s "DY" stride r j*normalized s stride r j
def biasTerm (s : State ℝ) (stride j : Nat) (r : Nat) : ℝ :=
  SourceGroupedReal.row s "DY" stride r j

theorem terms_after_store (m : Model ℝ) (s : State ℝ) (stride N B cur j : Nat) :
    weightTerm (storeDX m s stride N B cur) stride j=weightTerm s stride j ∧
      biasTerm (storeDX m s stride N B cur) stride j=biasTerm s stride j := by
  constructor <;> funext r
  · simp only [weightTerm,row_after_store m s stride N B cur r "DY" (by decide),normalized_after_store]
  · simp only [biasTerm,row_after_store m s stride N B cur r "DY" (by decide)]

theorem fold_tail (m : Model ℝ) (laws : SourceForwardReal.Laws m) (stride M N G B group cur : Nat)
    (s : State ℝ) (a : Carries ℝ B) (hg : 0<G) (hcur : group<cur)
    (i : TileIndex [B]) (hi : i.1.val<N) :
    (fold m stride M N G B group cur s a).2.1 i=
      a.1 i+((SourceRows.rows M G cur).map (weightTerm s stride i.1.val)).sum ∧
    (fold m stride M N G B group cur s a).2.2 i=
      a.2 i+((SourceRows.rows M G cur).map (biasTerm s stride i.1.val)).sum := by
  rw [fold,SourceRows.rows]
  simp only [Nat.ne_of_gt hg,if_false]
  split
  · rename_i hc
    have ht := fold_tail m laws stride M N G B group (cur+G) (storeDX m s stride N B cur)
      (nextCarries m s stride N B group cur a) hg (by omega) i hi
    have hn := nextCarries_real m laws s stride N B group cur a i hi
    rw [ht.1,ht.2,hn.1,hn.2,(terms_after_store m s stride N B cur i.1.val).1,
      (terms_after_store m s stride N B cur i.1.val).2]
    simp only [show cur≠group by omega,if_false,List.map_cons,List.sum_cons,weightTerm,biasTerm]
    constructor <;> ring
  · rename_i hc
    simp [hc]
termination_by M-cur
decreasing_by omega

theorem fold_group (m : Model ℝ) (laws : SourceForwardReal.Laws m) (stride M N G B group : Nat)
    (s : State ℝ) (hg : 0<G) (i : TileIndex [B]) (hi : i.1.val<N) :
    (fold m stride M N G B group group s (zero m B)).2.1 i=
      ((groupRows M G group).map (weightTerm s stride i.1.val)).sum ∧
    (fold m stride M N G B group group s (zero m B)).2.2 i=
      ((groupRows M G group).map (biasTerm s stride i.1.val)).sum := by
  rw [←SourceRows.rows_eq M G group hg,fold,SourceRows.rows]
  simp only [Nat.ne_of_gt hg,if_false]
  split
  · rename_i hc
    have ht := fold_tail m laws stride M N G B group (group+G) (storeDX m s stride N B group)
      (nextCarries m s stride N B group group (zero m B)) hg (by omega) i hi
    have hn := nextCarries_real m laws s stride N B group group (zero m B) i hi
    rw [ht.1,ht.2,hn.1,hn.2,(terms_after_store m s stride N B group i.1.val).1,
      (terms_after_store m s stride N B group i.1.val).2]
    simp [weightTerm,biasTerm]
  · rename_i hc
    simp [hc,zero,laws.literal]

#axiomsClean fold_group
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceGroupedSums
