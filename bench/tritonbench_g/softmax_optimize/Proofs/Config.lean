import Mathlib.Data.Nat.Log
import Mathlib.Tactic
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.Config

/-- Python `min(4096, 1 << (N.bit_length()-1))`, used only after N>0. -/
def tile (N : Nat) : Nat := min 4096 (2 ^ N.log2)

theorem tile_positive (N : Nat) : 0 < tile N := by
  unfold tile
  exact lt_min (by norm_num) (pow_pos (by norm_num) _)

theorem tile_le_width {N : Nat} (hN : 0 < N) : tile N ≤ N := by
  have h : 2 ^ N.log2 ≤ N := by
    rw [Nat.log2_eq_log_two]
    exact Nat.pow_log_le_self 2 (by omega)
  exact le_trans (min_le_right _ _) h

theorem tile_le_cap (N : Nat) : tile N ≤ 4096 := min_le_left _ _

theorem tile_nonempty_source_domain {N : Nat} (hN : 0 < N) :
    0 < tile N ∧ tile N ≤ N := ⟨tile_positive N, tile_le_width hN⟩

def fullTiles (N B : Nat) : Nat := (N - 1) / B
def previous (N B : Nat) : Nat := (N + B - 1) / B * B - B

theorem previous_eq_full_tiles {N B : Nat} (hN : 0 < N) (hB : 0 < B) :
    previous N B = fullTiles N B * B := by
  unfold previous fullTiles
  rw [show N + B - 1 = (N - 1) + B by omega, Nat.add_div_right _ hB]
  simp [Nat.add_mul]

theorem tail_start_lt {N B : Nat} (hN : 0 < N) : fullTiles N B * B < N := by
  have h := Nat.div_mul_le_self (N - 1) B
  unfold fullTiles
  omega

theorem tail_covers {N B : Nat} (hN : 0 < N) (hB : 0 < B) :
    N ≤ (fullTiles N B + 1) * B := by
  have h := Nat.div_add_mod' (N - 1) B
  have hm := Nat.mod_lt (N - 1) hB
  unfold fullTiles
  rw [Nat.add_mul, Nat.one_mul]
  omega

theorem full_tile_active {N B tileIndex lane : Nat} (hN : 0 < N)
    (ht : tileIndex < fullTiles N B) (hl : lane < B) : tileIndex * B + lane < N := by
  have h := tail_start_lt (B := B) hN
  have htB := Nat.mul_le_mul_right B (show tileIndex + 1 ≤ fullTiles N B by omega)
  nlinarith

theorem final_tail_nonempty_once {N B : Nat} (hN : 0 < N) (hB : 0 < B) :
    previous N B < N ∧ N ≤ previous N B + B := by
  rw [previous_eq_full_tiles hN hB]
  constructor
  · exact tail_start_lt hN
  · simpa [Nat.add_mul] using tail_covers hN hB

/-- Empty shapes return the allocated empty tensor before any helper or launch. -/
def launches (M N : Nat) : Bool := decide (0 < M ∧ 0 < N)

theorem empty_rows_no_launch (N : Nat) : launches 0 N = false := by simp [launches]
theorem empty_columns_no_launch (M : Nat) : launches M 0 = false := by simp [launches]

theorem selected_three : tile 3 = 2 := by decide
theorem selected_4095 : tile 4095 = 2048 := by decide
theorem selected_4096 : tile 4096 = 4096 := by decide
theorem selected_4097 : tile 4097 = 4096 := by decide

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SoftmaxOptimize.Config
