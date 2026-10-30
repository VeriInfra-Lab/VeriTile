import bench.repairs.tritonbench_g.attention_llama.Proofs.TailMath
import bench.repairs.tritonbench_g.attention_llama.Proofs.Geometry

namespace VeriTile.Bench.Repairs.AttentionLlama.MaskCoverage
open TailMath

theorem processed_eq (N : Nat) (causal : Bool) (position query n : Nat) :
    processed (blockKeys N causal position query) n =
      (logicalKeys N causal position query).filter (fun key => key < n * 64) := by
  induction n with
  | zero => simp [processed]
  | succ n ih =>
    change processed (blockKeys N causal position query) n ∪
      blockKeys N causal position query n = _
    rw [ih]
    ext key
    simp only [Finset.mem_union, blockKeys, Finset.mem_filter]
    by_cases hk : key ∈ logicalKeys N causal position query
    · simp only [hk, true_and]
      omega
    · simp only [hk, false_and, or_self]

theorem blocks_disjoint (N : Nat) (causal : Bool) (position query n : Nat) :
    Disjoint (processed (blockKeys N causal position query) n)
      (blockKeys N causal position query n) := by
  rw [processed_eq, Finset.disjoint_left]
  intro key hk hb
  simp only [Finset.mem_filter, blockKeys] at hk hb
  omega

theorem visible_before_stop (N : Nat) (causal : Bool) (position pid lane key : Nat)
    (hl : lane < 64) (hk : key ∈ logicalKeys N causal position (pid * 64 + lane)) :
    key < stop N causal position pid := by
  simp only [logicalKeys, Finset.mem_filter, Finset.mem_range, visible] at hk
  cases causal <;> simp only [stop, Bool.false_eq_true, ↓reduceIte]
  · exact hk.1
  · simp only [Bool.true_eq_false, false_or] at hk
    omega

theorem all_keys_covered (N : Nat) (causal : Bool) (position pid lane : Nat)
    (hl : lane < 64) :
    processed (blockKeys N causal position (pid * 64 + lane))
      (count (stop N causal position pid)) =
      logicalKeys N causal position (pid * 64 + lane) := by
  rw [processed_eq]
  apply Finset.filter_eq_self.mpr
  intro key hk
  exact Nat.lt_of_lt_of_le (visible_before_stop N causal position pid lane key hl hk)
    (count_covers _)

theorem row_result (score value : Nat → ℝ) (select : Nat → ℝ)
    (N : Nat) (causal : Bool) (position pid lane : Nat) (hl : lane < 64) :
    (run score value select (blockKeys N causal position (pid * 64 + lane))
      (count (stop N causal position pid))).accumulator =
      expected score value (logicalKeys N causal position (pid * 64 + lane)) := by
  rw [run_eq_weighted_sum score value select _ _
    (fun t _ => blocks_disjoint N causal position (pid * 64 + lane) t)]
  rw [all_keys_covered N causal position pid lane hl]


end VeriTile.Bench.Repairs.AttentionLlama.MaskCoverage
