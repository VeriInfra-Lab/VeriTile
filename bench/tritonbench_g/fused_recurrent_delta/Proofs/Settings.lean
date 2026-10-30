module
public import Init
import all Init.Data.Nat.Power2.Basic
public section
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.Settings
theorem nextPower_covers (n : Nat) : n≤n.nextPowerOfTwo := by
  apply go 1 (by decide)
where
  go (power : Nat) (hp : 0<power) : n≤Nat.nextPowerOfTwo.go n power hp := by
    unfold Nat.nextPowerOfTwo.go
    split
    · exact go (power*2) (Nat.mul_pos hp (by decide))
    · apply Nat.le_of_not_gt; assumption
  termination_by n-power
  decreasing_by simp_wf; apply Nat.nextPowerOfTwo_dec <;> assumption
end VeriTile.Bench.Repairs.FusedRecurrentDelta.Settings
