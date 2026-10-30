import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.FP32Prior.Proofs.Scatter
import Mathlib
namespace VeriTile.Bench.Repairs.DequantizeRowwise.FP32Prior.Bytes
abbrev Byte := BitVec 8
abbrev Word := Fin 4 → Byte
abbrev Memory := Nat → Byte
abbrev Coord (rows columns : Nat) := Fin rows × Fin columns

def linear {rows columns : Nat} (q : Coord rows columns) := q.1.val*columns+q.2.val

theorem linear_bound {rows columns : Nat} (q : Coord rows columns) : linear q<rows*columns := by
  have h := q.2.isLt
  calc
    linear q < (q.1.val+1)*columns := by unfold linear;simp only [Nat.add_mul,Nat.one_mul];omega
    _ ≤ rows*columns := Nat.mul_le_mul_right _ (by have := q.1.isLt;omega)

theorem linear_injective {rows columns : Nat} : Function.Injective (@linear rows columns) := by
  intro a b eq
  have ha := a.2.isLt
  have hb := b.2.isLt
  have positive : 0<columns := by omega
  have rowsEq := congrArg (fun n=>n/columns) eq
  have hr : a.1.val=b.1.val := by
    simpa [linear,Nat.mul_comm a.1.val columns,Nat.mul_comm b.1.val columns,Nat.mul_add_div positive,Nat.div_eq_of_lt ha,Nat.div_eq_of_lt hb] using rowsEq
  apply Prod.ext (Fin.ext hr)
  apply Fin.ext
  unfold linear at eq
  rw [hr] at eq
  omega


end VeriTile.Bench.Repairs.DequantizeRowwise.FP32Prior.Bytes
