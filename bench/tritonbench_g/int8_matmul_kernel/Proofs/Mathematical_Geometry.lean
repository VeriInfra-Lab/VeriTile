import Mathlib.Tactic
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
def groupBase (rows cols gm pid : Nat) := pid / (gm*cols)*gm
def groupSize (rows cols gm pid : Nat) := min (rows-groupBase rows cols gm pid) gm
def tileRow (rows cols gm pid : Nat) :=
  groupBase rows cols gm pid + (pid%(gm*cols))%groupSize rows cols gm pid
def tileCol (rows cols gm pid : Nat) := (pid%(gm*cols))/groupSize rows cols gm pid

theorem grouped_bounds (rows cols gm pid : Nat) (hgm : 0<gm) (hp : pid<rows*cols) :
    0<groupSize rows cols gm pid ∧ tileRow rows cols gm pid<rows ∧ tileCol rows cols gm pid<cols := by
  have hc : 0<cols := by nlinarith
  have hmul := Nat.div_mul_le_self pid (gm*cols)
  have hb : groupBase rows cols gm pid<rows := by unfold groupBase; nlinarith
  have hs : 0<groupSize rows cols gm pid := by unfold groupSize; omega
  have hspan : groupBase rows cols gm pid+groupSize rows cols gm pid≤rows := by
    have := min_le_left (rows-groupBase rows cols gm pid) gm
    unfold groupSize
    omega
  have hrem : pid%(gm*cols)<groupSize rows cols gm pid*cols := by
    unfold groupSize
    by_cases hsmall : rows-groupBase rows cols gm pid≤gm
    · rw [min_eq_left hsmall]
      have hd := Nat.mod_add_div pid (gm*cols)
      have hsub := Nat.sub_add_cancel (Nat.le_of_lt hb)
      unfold groupBase at hsub ⊢
      nlinarith
    · rw [min_eq_right (by omega)]
      exact Nat.mod_lt _ (Nat.mul_pos hgm hc)
  refine ⟨hs,?_,?_⟩
  · have := Nat.mod_lt (pid%(gm*cols)) hs
    unfold tileRow
    omega
  · unfold tileCol
    exact (Nat.div_lt_iff_lt_mul hs).mpr (by simpa [Nat.mul_comm] using hrem)

theorem grouped_injective (rows cols gm a b : Nat)
    (hgm : 0<gm) (ha : a<rows*cols) (hb : b<rows*cols)
    (hr : tileRow rows cols gm a=tileRow rows cols gm b)
    (hc : tileCol rows cols gm a=tileCol rows cols gm b) : a=b := by
  have group_of_row (pid : Nat) (hp : pid<rows*cols) :
      tileRow rows cols gm pid/gm=pid/(gm*cols) := by
    have hs := (grouped_bounds rows cols gm pid hgm hp).1
    have hm := Nat.mod_lt (pid%(gm*cols)) hs
    have hle : groupSize rows cols gm pid≤gm := min_le_right _ _
    unfold tileRow groupBase
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hgm]
    rw [Nat.div_eq_of_lt (show (pid%(gm*cols))%groupSize rows cols gm pid<gm by omega)]
    simp
  have hq : a/(gm*cols)=b/(gm*cols) := by rw [←group_of_row a ha,←group_of_row b hb,hr]
  have hbase : groupBase rows cols gm a=groupBase rows cols gm b := by simp [groupBase,hq]
  have hsize : groupSize rows cols gm a=groupSize rows cols gm b := by simp [groupSize,hbase]
  have hm : (a%(gm*cols))%groupSize rows cols gm a=(b%(gm*cols))%groupSize rows cols gm a := by
    unfold tileRow at hr
    rw [←hbase,←hsize] at hr
    exact Nat.add_left_cancel hr
  unfold tileCol at hc
  change a%(gm*cols)/groupSize rows cols gm a=b%(gm*cols)/groupSize rows cols gm b at hc
  rw [←hsize] at hc
  exact Nat.ext_div_mod hq (Nat.ext_div_mod hc hm)

def coordinate (rows cols gm : Nat) (hgm : 0<gm) (p : Fin (rows*cols)) : Fin rows × Fin cols :=
  (⟨tileRow rows cols gm p.val,(grouped_bounds rows cols gm p.val hgm p.isLt).2.1⟩,
   ⟨tileCol rows cols gm p.val,(grouped_bounds rows cols gm p.val hgm p.isLt).2.2⟩)

theorem coordinate_bijective (rows cols gm : Nat) (hgm : 0<gm) : Function.Bijective (coordinate rows cols gm hgm) := by
  apply (Fintype.bijective_iff_injective_and_card _).mpr
  refine ⟨?_,by simp⟩
  intro a b he
  apply Fin.ext
  exact grouped_injective rows cols gm a.val b.val hgm a.isLt b.isLt
    (congrArg (fun x => x.1.val) he) (congrArg (fun x => x.2.val) he)

theorem grouped_covered (rows cols gm i j : Nat) (hgm : 0<gm) (hi : i<rows) (hj : j<cols) :
    ∃ pid, pid<rows*cols ∧ tileRow rows cols gm pid=i ∧ tileCol rows cols gm pid=j := by
  obtain ⟨p,hp⟩ := (coordinate_bijective rows cols gm hgm).2 (⟨i,hi⟩,⟨j,hj⟩)
  exact ⟨p.val,p.isLt,congrArg (fun x => x.1.val) hp,congrArg (fun x => x.2.val) hp⟩


#axiomsClean grouped_bounds
#axiomsClean grouped_injective
#axiomsClean coordinate_bijective
#axiomsClean grouped_covered
end
end VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
