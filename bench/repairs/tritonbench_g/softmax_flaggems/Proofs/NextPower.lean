module
import all Init.Data.Nat.Power2.Basic
import Init.Data.Nat.Lemmas
import Init.Omega
public section
namespace VeriTile.Bench.Repairs.SoftmaxFlaggems.Integer
private theorem go_bound (n p : Nat) (hp : 0<p) (hn : n≤32768) (hb : p<2^17) :
    Nat.nextPowerOfTwo.go n p hp<2^17 := by
  unfold Nat.nextPowerOfTwo.go
  split
  · rename_i h
    exact go_bound n (p*2) (Nat.mul_pos hp (by decide)) hn (by omega)
  · exact hb
termination_by n-p
decreasing_by apply Nat.nextPowerOfTwo_dec <;> assumption

theorem next_power_bound (n : Nat) (h : n≤32768) : n.nextPowerOfTwo<2^17 :=
  go_bound n 1 (by decide) h (by decide)
end VeriTile.Bench.Repairs.SoftmaxFlaggems.Integer
