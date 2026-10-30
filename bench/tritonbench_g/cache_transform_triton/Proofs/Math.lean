import bench.tritonbench_g.cache_transform_triton.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.CacheTransform
open VeriTile.Triton

inductive Mode where
  | prompt | decode
  deriving DecidableEq

/-- Host implementation: exact natural row indices after signed validation. -/
def preparedRows : Mode → List Nat → List Nat
  | .prompt, lengths => lengths.flatMap List.range
  | .decode, lengths => lengths.map (· - 1)

/-- Independent tensor mathematics: concatenate each sequence's finite row
vector, or take the last row of every positive sequence. No kernel state,
address, metadata buffer, program ID or prepared row table occurs here. -/
def mathematical {α : Type} (mode : Mode) (lengths : List Nat) (cache : Nat → α) : List α :=
  match mode with
  | .prompt => (lengths.map (fun n => List.ofFn (fun r : Fin n => cache r.val))).flatten
  | .decode => lengths.map (fun n => cache (n - 1))

theorem map_range_ofFn {α : Type} (n : Nat) (f : Nat → α) :
    (List.range n).map f = List.ofFn (fun i : Fin n => f i.val) := by
  apply List.ext_getElem
  · simp
  · intro i h₁ h₂
    simp

theorem preparedRows_math {α : Type} (mode : Mode) (lengths : List Nat) (cache : Nat → α) :
    (preparedRows mode lengths).map cache = mathematical mode lengths cache := by
  cases mode
  · simp only [preparedRows, mathematical]
    rw [List.map_flatMap]
    simp only [map_range_ofFn]
    rfl
  · simp [preparedRows, mathematical, List.map_map]

theorem preparedRows_length (mode : Mode) (lengths : List Nat) :
    (preparedRows mode lengths).length =
      match mode with | .prompt => lengths.sum | .decode => lengths.length := by
  cases mode
  · induction lengths with
    | nil => rfl
    | cons n ns ih => simpa [preparedRows] using congrArg (n + ·) ih
  · simp [preparedRows]

def ValidLengths (mode : Mode) (lengths : List Nat) (capacity : Nat) : Prop :=
  ∀ n ∈ lengths, n ≤ capacity ∧ (mode = .decode → 0 < n)

theorem preparedRows_bound (mode : Mode) (lengths : List Nat) (capacity : Nat)
    (h : ValidLengths mode lengths capacity) :
    ∀ row ∈ preparedRows mode lengths, row < capacity := by
  intro row hr
  cases mode
  · obtain ⟨n, hn, hr⟩ := List.mem_flatMap.mp hr
    exact lt_of_lt_of_le (List.mem_range.mp hr) (h n hn).1
  · obtain ⟨n, hn, rfl⟩ := List.mem_map.mp hr
    have := h n hn
    have := this.2 rfl
    omega

/-- Signed int32/int64 metadata is checked before conversion/subtraction.
Python integer operations coincide with these exact Int operations. -/
theorem validated_signed_length (v : Int) (cap : Nat) (h₀ : 0 ≤ v)
    (hcap : v ≤ cap) : v.toNat ≤ cap ∧ (v.toNat : Int) = v := by
  constructor
  · omega
  · exact Int.toNat_of_nonneg h₀

theorem decode_subtract (v : Int) (h : 1 ≤ v) :
    (v - 1).toNat = v.toNat - 1 := by omega

#axiomsClean preparedRows_math
#axiomsClean preparedRows_bound
end VeriTile.Bench.Repairs.CacheTransform
