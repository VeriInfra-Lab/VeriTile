import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.ReturnedStorage
import Mathlib.Data.Nat.ModEq

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.ViewGuard
open ReturnedStorage

/-- Exactly the repaired wrapper's positive-dimension stride checks. -/
structure Admitted (v : View) : Prop where
  row : 1<v.rows → v.rowStride≠0
  column : 1<v.columns → v.columnStride≠0
  gcd : 1<v.rows → 1<v.columns →
    v.columns≤v.rowStride/Nat.gcd v.rowStride v.columnStride ∨
    v.rows≤v.columnStride/Nat.gcd v.rowStride v.columnStride

theorem writable (v : View) (guard : Admitted v) : v.Writable := by
  intro i j same
  have hi := i.1.isLt
  have hj := j.1.isLt
  have hc := i.2.isLt
  have hd := j.2.isLt
  unfold View.offset at same
  have equal : i.1.val*v.rowStride+i.2.val*v.columnStride=
      j.1.val*v.rowStride+j.2.val*v.columnStride := by omega
  by_cases rows : v.rows≤1
  · have first : i.1=j.1 := Fin.ext (by omega)
    have second : i.2=j.2 := by
      by_cases cols : v.columns≤1
      · exact Fin.ext (by omega)
      · apply Fin.ext
        apply Nat.eq_of_mul_eq_mul_right (Nat.pos_of_ne_zero (guard.column (by omega)))
        simpa only [first,add_right_inj] using equal
    exact Prod.ext first second
  by_cases cols : v.columns≤1
  · have second : i.2=j.2 := Fin.ext (by omega)
    have first : i.1=j.1 := by
      apply Fin.ext
      apply Nat.eq_of_mul_eq_mul_right (Nat.pos_of_ne_zero (guard.row (by omega)))
      simpa only [second,add_left_inj] using equal
    exact Prod.ext first second
  rcases guard.gcd (by omega) (by omega) with bound | bound
  · have mod : Nat.ModEq v.rowStride (i.2.val*v.columnStride) (j.2.val*v.columnStride) := by
      have := congrArg (fun n => n%v.rowStride) equal
      simpa [Nat.add_mod] using this
    have cancelled := mod.cancel_right_div_gcd (Nat.pos_of_ne_zero (guard.row (by omega)))
    have second : i.2=j.2 := by
      apply Fin.ext
      change i.2.val%(v.rowStride/Nat.gcd v.rowStride v.columnStride)=
        j.2.val%(v.rowStride/Nat.gcd v.rowStride v.columnStride) at cancelled
      simpa only [Nat.mod_eq_of_lt (hc.trans_le bound),Nat.mod_eq_of_lt (hd.trans_le bound)] using cancelled
    have first : i.1=j.1 := by
      apply Fin.ext
      apply Nat.eq_of_mul_eq_mul_right (Nat.pos_of_ne_zero (guard.row (by omega)))
      simpa only [second,add_left_inj] using equal
    exact Prod.ext first second
  · have mod : Nat.ModEq v.columnStride (i.1.val*v.rowStride) (j.1.val*v.rowStride) := by
      have := congrArg (fun n => n%v.columnStride) equal
      simpa [Nat.add_mod] using this
    have cancelled := mod.cancel_right_div_gcd (Nat.pos_of_ne_zero (guard.column (by omega)))
    have first : i.1=j.1 := by
      apply Fin.ext
      change i.1.val%(v.columnStride/Nat.gcd v.columnStride v.rowStride)=
        j.1.val%(v.columnStride/Nat.gcd v.columnStride v.rowStride) at cancelled
      rw [Nat.gcd_comm] at cancelled
      simpa only [Nat.mod_eq_of_lt (hi.trans_le bound),Nat.mod_eq_of_lt (hj.trans_le bound)] using cancelled
    have second : i.2=j.2 := by
      apply Fin.ext
      apply Nat.eq_of_mul_eq_mul_right (Nat.pos_of_ne_zero (guard.column (by omega)))
      simpa only [first,add_right_inj] using equal
    exact Prod.ext first second

#axiomsClean writable
end VeriTile.Bench.Repairs.BgmvExpandSlice.ViewGuard
