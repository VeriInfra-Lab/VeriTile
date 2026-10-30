import VeriTile.Triton.Random.Values
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Tactic

/-!
Independent mathematical specification for the complete one-pass forward
kernel. All values come from the input snapshot, including both seed streams.
These definitions are not an execution theorem or an FP equivalence theorem.
The generator is the canonical seven-round Philox generator, not a callback.
-/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd

open VeriTile.Triton
noncomputable section

structure Flags where
  rms : Bool
  residual : Bool
  residualOut : Bool
  bias : Bool
  dropout : Bool
  dropoutMask : Bool
  rowscale : Bool
  x1 : Bool
  w1 : Bool
  b1 : Bool
  deriving DecidableEq

/-- The source memory before execution. The second rowscale and seed stream
begins at M, exactly as in the JIT. Rows are indexed independently of strides;
the source contract supplies their actual memory addresses. -/
structure Inputs where
  x : Nat → Nat → ℝ
  x1 : Nat → Nat → ℝ
  residual : Nat → Nat → ℝ
  weight : Nat → ℝ
  bias : Nat → ℝ
  weight1 : Nat → ℝ
  bias1 : Nat → ℝ
  rowscale : Nat → ℝ
  seeds : Nat → Int

namespace Formula

/-- Exact unsigned cast of the loaded int64 seed. This explicitly retains
the source narrowing cast; replacing the pointer ABI by uint32 is invalid. -/
def seedU32 (seed : Int) : Int := seed % (2^32 : Int)

def keep (seed : Int) (p : ℝ) (col : Nat) : Bool :=
  decide (Random.randReal (seedU32 seed) (col : Int) .bits32 7 > p)

def dropout (seed : Int) (p value : ℝ) (col : Nat) : ℝ :=
  if keep seed p col then value / (1-p) else 0

def stream (f : Flags) (a : Inputs) (M : Nat) (p : ℝ)
    (second : Bool) (row col : Nat) : ℝ :=
  let input := if second then a.x1 row col else a.x row col
  let streamRow := if second then M + row else row
  let scaled := if f.rowscale then input * a.rowscale streamRow else input
  if f.dropout then dropout (a.seeds streamRow) p scaled col else scaled

/-- Complete residual-input value after row scaling, dropout, X1 and residual.
The order of the additions matches the source. -/
def combined (f : Flags) (a : Inputs) (M : Nat) (p : ℝ) (row col : Nat) : ℝ :=
  let main := stream f a M p false row col
  let twoInputs := if f.x1 then main + stream f a M p true row col else main
  if f.residual then twoInputs + a.residual row col else twoInputs

def mean (f : Flags) (a : Inputs) (M N : Nat) (p : ℝ) (row : Nat) : ℝ :=
  (∑ col ∈ Finset.range N, combined f a M p row col) / N

def centered (f : Flags) (a : Inputs) (M N : Nat) (p : ℝ) (row col : Nat) : ℝ :=
  if f.rms then combined f a M p row col
  else combined f a M p row col - mean f a M N p row

def variance (f : Flags) (a : Inputs) (M N : Nat) (p : ℝ) (row : Nat) : ℝ :=
  (∑ col ∈ Finset.range N, (centered f a M N p row col)^2) / N

def rstd (f : Flags) (a : Inputs) (M N : Nat) (eps p : ℝ) (row : Nat) : ℝ :=
  1 / Real.sqrt (variance f a M N p row + eps)

def normalized (f : Flags) (a : Inputs) (M N : Nat) (eps p : ℝ)
    (row col : Nat) : ℝ :=
  centered f a M N p row col * rstd f a M N eps p row

def y (f : Flags) (a : Inputs) (M N : Nat) (eps p : ℝ) (row col : Nat) : ℝ :=
  let weighted := normalized f a M N eps p row col * a.weight col
  if f.bias then weighted + a.bias col else weighted

def y1 (f : Flags) (a : Inputs) (M N : Nat) (eps p : ℝ) (row col : Nat) : ℝ :=
  let weighted := normalized f a M N eps p row col * a.weight1 col
  if f.b1 then weighted + a.bias1 col else weighted

theorem seedU32_range (seed : Int) : 0 ≤ seedU32 seed ∧ seedU32 seed < 2^32 := by
  exact ⟨Int.emod_nonneg seed (by norm_num), Int.emod_lt_of_pos seed (by norm_num)⟩

/-- The host's torch.randint(2**32, dtype=int64) domain makes the unsigned
cast value-preserving, while its source ABI remains int64. -/
theorem seedU32_of_host_range (seed : Int) (h : 0 ≤ seed ∧ seed < 2^32) :
    seedU32 seed = seed := Int.emod_eq_of_lt h.1 h.2

theorem dropout_zero (seed : Int) (p : ℝ) (col : Nat) :
    dropout seed p 0 col = 0 := by simp [dropout]

theorem dropout_keep (seed : Int) (p value : ℝ) (col : Nat)
    (h : keep seed p col = true) : dropout seed p value col = value / (1-p) := by
  simp [dropout, h]

theorem dropout_drop (seed : Int) (p value : ℝ) (col : Nat)
    (h : keep seed p col = false) : dropout seed p value col = 0 := by
  simp [dropout, h]

/-- Source padding occurs before scaling and dropout. This snapshot makes
the inactive-lane obligation explicit, including both input streams. -/
def paddedInputs (a : Inputs) (N : Nat) : Inputs :=
  { a with
    x := fun row col => if col < N then a.x row col else 0
    x1 := fun row col => if col < N then a.x1 row col else 0
    residual := fun row col => if col < N then a.residual row col else 0 }

theorem stream_padded (f : Flags) (a : Inputs) (M N : Nat) (p : ℝ)
    (second : Bool) (row col : Nat) :
    stream f (paddedInputs a N) M p second row col =
      if col < N then stream f a M p second row col else 0 := by
  by_cases h : col < N
  · simp [stream, paddedInputs, h]
  · cases second <;> cases hscale : f.rowscale <;> cases hdrop : f.dropout
    all_goals simp [stream, paddedInputs, h, hscale, hdrop, dropout_zero]

theorem combined_padded (f : Flags) (a : Inputs) (M N : Nat) (p : ℝ)
    (row col : Nat) :
    combined f (paddedInputs a N) M p row col =
      if col < N then combined f a M p row col else 0 := by
  unfold combined
  rw [stream_padded, stream_padded]
  by_cases h : col < N
  · simp [paddedInputs, h]
  · cases hx : f.x1 <;> cases hr : f.residual
    all_goals simp [paddedInputs, h]

/-- Every padded reduction has the logical N-column sum, at arbitrary legal
block size. No precomputed mean or variance is used. -/
theorem sum_padded (values : Nat → ℝ) {N block : Nat} (h : N ≤ block) :
    (∑ col ∈ Finset.range block, if col < N then values col else 0) =
      ∑ col ∈ Finset.range N, values col := by
  have hs := Finset.sum_subset (Finset.range_mono h)
    (f := fun col => if col < N then values col else (0 : ℝ))
    (by intro col _ hn; simp only [Finset.mem_range] at hn; simp [hn])
  rw [← hs]
  exact Finset.sum_congr rfl (by intro col hc; simp [Finset.mem_range.mp hc])

theorem sum_combined_padded (f : Flags) (a : Inputs) (M N block : Nat)
    (p : ℝ) (row : Nat) (h : N ≤ block) :
    (∑ col ∈ Finset.range block, combined f (paddedInputs a N) M p row col) =
      ∑ col ∈ Finset.range N, combined f a M p row col := by
  simp_rw [combined_padded]
  exact sum_padded _ h

theorem variance_nonneg (f : Flags) (a : Inputs) (M N : Nat) (p : ℝ) (row : Nat) :
    0 ≤ variance f a M N p row := by
  unfold variance
  apply div_nonneg
  · exact Finset.sum_nonneg (by intro col _; exact sq_nonneg _)
  · exact Nat.cast_nonneg _

theorem rstd_pos (f : Flags) (a : Inputs) (M N : Nat) (eps p : ℝ) (row : Nat)
    (heps : 0 < eps) : 0 < rstd f a M N eps p row := by
  unfold rstd
  apply one_div_pos.mpr
  apply Real.sqrt_pos.mpr
  exact add_pos_of_nonneg_of_pos (variance_nonneg f a M N p row) heps

end Formula
end
end VeriTile.Bench.TritonBenchG.LayerNormFwd
