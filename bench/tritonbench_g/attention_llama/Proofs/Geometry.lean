import Mathlib.Tactic

/-! Shared integer geometry; independent of the real and FP value proofs. -/
namespace VeriTile.Bench.Repairs.AttentionLlama.MaskCoverage

def visible (causal : Bool) (position query key : Nat) : Prop :=
  causal = false ∨ key + position ≤ query

instance (causal : Bool) (position query key : Nat) :
    Decidable (visible causal position query key) := inferInstanceAs (Decidable (_ ∨ _))

def logicalKeys (N : Nat) (causal : Bool) (position query : Nat) : Finset Nat :=
  (Finset.range N).filter (visible causal position query)

def blockKeys (N : Nat) (causal : Bool) (position query block : Nat) : Finset Nat :=
  (logicalKeys N causal position query).filter
    (fun key => block * 64 ≤ key ∧ key < (block + 1) * 64)

def count (stop : Nat) : Nat := (stop + 63) / 64

def stop (N : Nat) (causal : Bool) (position pid : Nat) : Nat :=
  if causal then (pid + 1) * 64 + position else N

theorem count_covers (n : Nat) : n ≤ count n * 64 := by
  have h := Nat.div_add_mod (n + 63) 64
  have hm := Nat.mod_lt (n + 63) (by decide : 0 < 64)
  simp only [count]
  omega

theorem query_program_exists (H query : Nat) (hq : query < H) :
    query / 64 < count H ∧ query % 64 < 64 ∧
      query / 64 * 64 + query % 64 = query := by
  have hc := count_covers H
  have hm := Nat.mod_lt query (by decide : 0 < 64)
  have hd := Nat.div_add_mod query 64
  constructor
  · omega
  · constructor
    · exact hm
    · omega

theorem query_program_unique (query pid lane : Nat) (hl : lane < 64)
    (he : pid * 64 + lane = query) : pid = query / 64 ∧ lane = query % 64 := by
  subst query
  constructor <;> omega

theorem batch_head_roundtrip (heads batch head : Nat) (hh : head < heads) :
    (batch * heads + head) / heads = batch ∧
      (batch * heads + head) % heads = head := by
  have hp : 0 < heads := Nat.zero_lt_of_lt hh
  constructor
  · exact Nat.div_eq_of_lt_le (by omega) (by nlinarith)
  · simp [Nat.add_mod, Nat.mod_eq_of_lt hh]

end VeriTile.Bench.Repairs.AttentionLlama.MaskCoverage
