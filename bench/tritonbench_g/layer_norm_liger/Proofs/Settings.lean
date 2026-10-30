module
public import Mathlib.Tactic
import all Init.Data.Nat.Power2.Basic

public section
namespace VeriTile.Bench.Repairs.LayerNormLiger.Settings

/-- Reused natural-number fact from the fast-layernorm settings proof. -/
theorem nextPower_covers (N : Nat) : N ≤ N.nextPowerOfTwo := by
  apply go 1 (by decide)
where
  go (power : Nat) (hp : 0 < power) : N ≤ Nat.nextPowerOfTwo.go N power hp := by
    unfold Nat.nextPowerOfTwo.go
    split
    · exact go (power*2) (Nat.mul_pos hp (by decide))
    · omega
  termination_by N-power
  decreasing_by simp_wf; apply Nat.nextPowerOfTwo_dec <;> assumption

private theorem go_lower (N power : Nat) (hp : 0 < power) :
    power ≤ Nat.nextPowerOfTwo.go N power hp := by
  unfold Nat.nextPowerOfTwo.go
  split
  · exact le_trans (by omega) (go_lower N (power*2) (Nat.mul_pos hp (by decide)))
  · exact le_rfl
termination_by N-power
decreasing_by simp_wf; apply Nat.nextPowerOfTwo_dec <;> assumption

private theorem go_mono (N M power : Nat) (hp : 0 < power) (h : N ≤ M) :
    Nat.nextPowerOfTwo.go N power hp ≤ Nat.nextPowerOfTwo.go M power hp := by
  conv_lhs => unfold Nat.nextPowerOfTwo.go
  conv_rhs => unfold Nat.nextPowerOfTwo.go
  split_ifs with hn hm
  · exact go_mono N M (power*2) (Nat.mul_pos hp (by decide)) h
  · omega
  · exact le_trans (show power ≤ power*2 by omega) (go_lower M (power*2) (Nat.mul_pos hp (by decide)))
  · exact le_rfl
termination_by M-power
decreasing_by simp_wf; apply Nat.nextPowerOfTwo_dec <;> omega

theorem nextPower_bound (N : Nat) (h : N ≤ 65536) : N.nextPowerOfTwo ≤ 65536 := by
  have bound := go_mono N 65536 1 (by decide) h
  change N.nextPowerOfTwo ≤ Nat.nextPowerOfTwo 65536 at bound
  have maximum : Nat.nextPowerOfTwo 65536 = 65536 := by
    unfold Nat.nextPowerOfTwo
    iterate 17 (rw [Nat.nextPowerOfTwo.go]; norm_num)
  simpa only [maximum] using bound

def block (N : Nat) : Nat := N.nextPowerOfTwo

def warps (hip : Bool) (N : Nat) : Nat :=
  if block N ≥ 32768 then if hip then 16 else 32
  else if block N ≥ 8192 then 16
  else if block N ≥ 2048 then 8 else 4

end VeriTile.Bench.Repairs.LayerNormLiger.Settings
