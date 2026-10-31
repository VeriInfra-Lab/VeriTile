import Mathlib.Algebra.Order.Floor.Div
import Mathlib.Tactic
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.SourceArithmetic
set_option maxHeartbeats 300000

theorem ceiling (x b : Nat) (hb : 0<b) :
    x/b+(if x%b=0 then 0 else 1)=(x+b-1)/b := by
  have split : x+b-1=b*(x/b)+(x%b+b-1) := by
    have h:=Nat.mod_add_div x b
    omega
  rw [split,Nat.mul_add_div hb]
  by_cases hz:x%b=0
  · simp only [hz,↓reduceIte,Nat.zero_add,Nat.add_zero]
    have hlt:b-1<b:=by omega
    simp [Nat.div_eq_of_lt hlt]
  · have hr:=Nat.mod_lt x hb
    have hge:b≤x%b+b-1:=by omega
    have hlt:x%b+b-1<2*b:=by omega
    have he:(x%b+b-1)/b=1:=Nat.div_eq_of_lt_le (by simpa using hge) (by simpa [Nat.mul_comm] using hlt)
    simp only [hz,↓reduceIte,he]

#axiomsClean ceiling
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.SourceArithmetic
