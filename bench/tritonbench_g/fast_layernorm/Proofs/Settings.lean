module
public import Init
import all Init.Data.Nat.Power2.Basic

public section
namespace VeriTile.Bench.Repairs.FastLayernorm.Settings

/-- Natural-number library fact. The private recursion is made visible only
in this core-only helper; its full object family is pinned by the runner. -/
theorem nextPower_covers (N : Nat) : N≤N.nextPowerOfTwo := by
  apply go 1 (by decide)
where
  go (power : Nat) (hp : 0<power) : N≤Nat.nextPowerOfTwo.go N power hp := by
    unfold Nat.nextPowerOfTwo.go
    split
    · exact go (power*2) (Nat.mul_pos hp (by decide))
    · apply Nat.le_of_not_gt; assumption
  termination_by N-power
  decreasing_by simp_wf; apply Nat.nextPowerOfTwo_dec <;> assumption

end VeriTile.Bench.Repairs.FastLayernorm.Settings
