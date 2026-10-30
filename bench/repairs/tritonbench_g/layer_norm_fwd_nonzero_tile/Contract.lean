import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.Admission
import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Kernels
import Mathlib.Data.Nat.Log

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile
open VeriTile.Triton

abbrev inputCheck := VeriTile.Bench.Repairs.LayerNormFwd.Admission.checkInputs
abbrev InputDomain := VeriTile.Bench.Repairs.LayerNormFwd.Admission.InputDomain
abbrev inputCheck_iff := VeriTile.Bench.Repairs.LayerNormFwd.Admission.checkInputs_iff

/-- The installed next_power_of_2 convention, including its zero result. -/
def nextPower (N : Nat) : Nat :=
  if N=0 then 0 else if N≤1 then 1 else 2^(Nat.log2 (N-1)+1)
def originalTile (N : Nat) : Nat := min 16384 (nextPower N)
def tile (N : Nat) : Nat := max 1 (originalTile N)
def widthCheck (N : Nat) : Bool := decide (N≤tile N)

theorem tile_positive (N : Nat) : 0<tile N := by unfold tile; omega
theorem tile_limit (N : Nat) : tile N≤16384 := by unfold tile originalTile; omega
theorem tile_zero : tile 0=1 := by simp [tile,originalTile,nextPower]
theorem width_zero : widthCheck 0=Bool.true := by simp [widthCheck]
theorem tile_unchanged (N : Nat) (h : N≠0) : tile N=originalTile N := by
  have positive : 0<nextPower N := by
    simp only [nextPower,if_neg h]
    split_ifs
    · omega
    · exact Nat.two_pow_pos _
  unfold tile originalTile
  omega
theorem tile_arange (N : Nat) : Random.ArangeSizeDomain (tile N) := by
  refine ⟨tile_positive N,by have := tile_limit N; omega,?_⟩
  by_cases hz : N=0
  · subst N; exact ⟨0,tile_zero⟩
  rw [tile_unchanged N hz]
  by_cases hc : 16384≤nextPower N
  · exact ⟨14,by simp [originalTile,Nat.min_eq_left hc]⟩
  · rw [originalTile,Nat.min_eq_right (by omega : nextPower N≤16384)]
    simp only [nextPower,if_neg hz]
    split_ifs
    · exact ⟨0,rfl⟩
    · exact ⟨Nat.log2 (N-1)+1,rfl⟩
theorem widthCheck_iff (N : Nat) : widthCheck N=Bool.true ↔ N≤tile N := by
  simp [widthCheck]
theorem oversize_rejected (N : Nat) (h : 16384<N) : widthCheck N=Bool.false := by
  have := tile_limit N
  simp [widthCheck,show ¬N≤tile N by omega]

#axiomsClean inputCheck_iff
#axiomsClean tile_arange
#axiomsClean tile_unchanged
#axiomsClean width_zero
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile
