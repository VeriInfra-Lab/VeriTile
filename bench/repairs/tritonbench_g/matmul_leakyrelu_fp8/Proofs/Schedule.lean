import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Types
import bench.repairs.tritonbench_g.matmul_persistent_triton.Proofs.Bounds

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8
open VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
noncomputable section

theorem Problem.group_positive (p : Problem) (hm : 0<p.M) : 0<p.group :=
  lt_min (by decide) (RealMath.blocks_positive p.M 128 hm (by decide))

/-- This source uses the local PID remainder before the row remainder, unlike
the persistent kernel. The original size/column bounds remain reusable. -/
theorem Problem.grouped_bounds (p : Problem) (hg : 0<p.group) (tile : Nat) (ht : tile<p.tiles) :
    0<p.groupSize tile ∧ p.tileRow tile<p.rows ∧ p.tileColumn tile<p.columns := by
  have old := Schedule.grouped_bounds p.rows p.columns p.group tile hg ht
  have span := Nat.min_le_left (p.rows-p.groupBase tile) p.group
  have base : p.groupBase tile<p.rows := by
    have hb := Nat.div_mul_le_self tile (p.group*p.columns)
    change tile/(p.group*p.columns)*p.group<p.rows
    change tile<p.rows*p.columns at ht
    nlinarith
  have rem := Nat.mod_lt (tile%(p.group*p.columns)) old.1
  change (tile%(p.group*p.columns))%p.groupSize tile<p.groupSize tile at rem
  refine ⟨old.1,?_,old.2.2⟩
  change p.groupBase tile+(tile%(p.group*p.columns))%p.groupSize tile<p.rows
  change p.groupSize tile≤p.rows-p.groupBase tile at span
  omega

theorem Problem.group_of_row (p : Problem) (hg : 0<p.group) (tile : Nat) (ht : tile<p.tiles) :
    p.tileRow tile/p.group=tile/(p.group*p.columns) := by
  have hb := p.grouped_bounds hg tile ht
  have rem : (tile%(p.group*p.columns))%p.groupSize tile<p.group :=
    lt_of_lt_of_le (Nat.mod_lt _ hb.1) (Nat.min_le_right _ _)
  unfold Problem.tileRow Problem.groupBase Schedule.groupBase
  rw [Nat.add_comm,Nat.add_mul_div_right _ _ hg,Nat.div_eq_of_lt rem,Nat.zero_add]

theorem Problem.grouped_injective (p : Problem) (hg : 0<p.group) (a b : Nat)
    (ha : a<p.tiles) (hb : b<p.tiles)
    (hr : p.tileRow a=p.tileRow b) (hc : p.tileColumn a=p.tileColumn b) : a=b := by
  have hq : a/(p.group*p.columns)=b/(p.group*p.columns) := by
    rw [← p.group_of_row hg a ha,← p.group_of_row hg b hb,hr]
  have base : p.groupBase a=p.groupBase b := by simp only [Problem.groupBase,Schedule.groupBase,hq]
  have size : p.groupSize a=p.groupSize b := by
    change min (p.rows-p.groupBase a) p.group=min (p.rows-p.groupBase b) p.group
    rw [base]
  have rem : (a%(p.group*p.columns))%p.groupSize a=(b%(p.group*p.columns))%p.groupSize a := by
    unfold Problem.tileRow at hr
    rw [← base,← size] at hr
    exact Nat.add_left_cancel hr
  change (a%(p.group*p.columns))/p.groupSize a=(b%(p.group*p.columns))/p.groupSize b at hc
  rw [← size] at hc
  exact Nat.ext_div_mod hq (Nat.ext_div_mod hc rem)

def Problem.coordinate (p : Problem) (hg : 0<p.group) (tile : Fin p.tiles) : Fin p.rows × Fin p.columns :=
  ⟨⟨p.tileRow tile.val,(p.grouped_bounds hg tile.val tile.isLt).2.1⟩,
   ⟨p.tileColumn tile.val,(p.grouped_bounds hg tile.val tile.isLt).2.2⟩⟩

theorem Problem.coordinate_bijective (p : Problem) (hg : 0<p.group) :
    Function.Bijective (p.coordinate hg) := by
  apply (Fintype.bijective_iff_injective_and_card _).mpr
  constructor
  · intro a b same
    apply Fin.ext
    exact p.grouped_injective hg _ _ a.isLt b.isLt
      (congrArg (fun x => x.1.val) same) (congrArg (fun x => x.2.val) same)
  · simp only [Fintype.card_prod,Fintype.card_fin,Problem.tiles]
    exact Fintype.card_fin _

theorem Problem.grouped_covered (p : Problem) (hg : 0<p.group) (row column : Nat)
    (hr : row<p.rows) (hc : column<p.columns) :
    ∃ tile,tile<p.tiles ∧ p.tileRow tile=row ∧ p.tileColumn tile=column := by
  obtain ⟨tile,ht⟩ := (p.coordinate_bijective hg).2 (⟨row,hr⟩,⟨column,hc⟩)
  exact ⟨tile.val,tile.isLt,congrArg (fun x => x.1.val) ht,congrArg (fun x => x.2.val) ht⟩

#axiomsClean Problem.coordinate_bijective
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8
