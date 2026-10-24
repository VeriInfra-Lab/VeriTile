import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-! Independent real decay formulas with the source explicit signed low32 lane
conversion. Subsequent signed differences remain Int expressions in this model;
no native all-arithmetic machine-width or floating-rounding claim is made. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.Decay
open VeriTile.Triton
noncomputable section

def exp2 (x : ℝ) : ℝ := Real.exp (x * Real.log 2)
def headBase (H head : Nat) : ℝ :=
  Real.log (1 - exp2 (-5 - (head % H : Nat))) / Real.log 2

def factor (base exponent : ℝ) : ℝ := exp2 (exponent * base)
/-- Independent signed interpretation of the source lane's low32 bits. -/
def index32 (lane : Nat) : Int :=
  let bits : Nat := lane % 2^32
  if 2*bits < 2^32 then (bits : Int) else (bits : Int) - 2^32

def query (base : ℝ) (row : Nat) : ℝ := factor base ((index32 row + 1 : Int) : ℝ)
def key (base : ℝ) (length row : Nat) : ℝ :=
  factor base (((length : Int) - index32 row - 1 : Int) : ℝ)
def score (base : ℝ) (row column : Nat) : ℝ :=
  if index32 column ≤ index32 row then
    factor base ((index32 row - index32 column : Int) : ℝ) else 0

/-- Historical ideal lane formula, retained explicitly as a mathematical model. -/
def idealQuery (base : ℝ) (row : Nat) : ℝ := factor base (row + 1)
def idealKey (base : ℝ) (length row : Nat) : ℝ := factor base ((length : ℝ) - row - 1)
def idealScore (base : ℝ) (row column : Nat) : ℝ :=
  if column ≤ row then factor base ((row : ℝ) - column) else 0

/-- Exact ordinary nonnegative signed32 domain; used only for an explanatory
corollary, never as a premise of the complete source contracts. -/
theorem index32_small (lane : Nat) (h : lane < 2^31) : index32 lane = (lane : Int) := by
  norm_num [index32] at *
  rw [Nat.mod_eq_of_lt (by omega)]
  split_ifs <;> omega

theorem wrapped_lane_witness : index32 2147483648 = -2147483648 := by
  norm_num [index32]

theorem wrapped_causal_witness (base : ℝ) : score base 2147483648 0 = 0 := by
  norm_num [score,index32]

theorem ordinary_query_agrees (base : ℝ) (row : Nat) (h : row < 2^31) :
    query base row = idealQuery base row := by
  simp [query,idealQuery,index32_small row h]

theorem ordinary_key_agrees (base : ℝ) (length row : Nat) (h : row < 2^31) :
    key base length row = idealKey base length row := by
  simp [key,idealKey,index32_small row h]

theorem ordinary_score_agrees (base : ℝ) (row column : Nat)
    (hr : row < 2^31) (hc : column < 2^31) :
    score base row column = idealScore base row column := by
  simp [score,idealScore,index32_small row hr,index32_small column hc]

#axiomsClean index32_small
#axiomsClean wrapped_lane_witness
#axiomsClean wrapped_causal_witness
#axiomsClean ordinary_query_agrees
#axiomsClean ordinary_key_agrees
#axiomsClean ordinary_score_agrees

def chunkLength (T BT NT chunk : Nat) : Nat :=
  if chunk = NT - 1 ∧ T % BT ≠ 0 then T % BT else BT

theorem factor_pos (base exponent : ℝ) : 0 < factor base exponent :=
  Real.exp_pos _

theorem key_signed (base : ℝ) (length row : Nat) :
    key base length row = factor base (((length : Int) - index32 row - 1 : Int) : ℝ) := by
  simp [key]

theorem score_signed (base : ℝ) (row column : Nat) :
    score base row column =
      if index32 column ≤ index32 row then
        factor base ((index32 row - index32 column : Int) : ℝ) else 0 := by
  rfl

theorem tail_offset_negative (length row : Nat) (h : length ≤ row) :
    (length : Int) - row - 1 < 0 := by omega

theorem complete_chunk (T BT NT chunk : Nat)
    (h : chunk ≠ NT - 1 ∨ T % BT = 0) : chunkLength T BT NT chunk = BT := by
  rcases h with h | h <;> simp [chunkLength,h]

theorem final_partial_chunk (T BT NT : Nat) (h : T % BT ≠ 0) :
    chunkLength T BT NT (NT - 1) = T % BT := by simp [chunkLength,h]

/-- The head rate is in (0,1) for every natural head index, so its logarithm
has a positive argument. No extra distributional assumption is needed. -/
theorem head_rate_positive (H head : Nat) : 0 < 1 - exp2 (-5 - (head % H : Nat)) := by
  have hlog : (0 : ℝ) < Real.log 2 := Real.log_pos (by norm_num)
  have hindex : (0 : ℝ) ≤ (head % H : Nat) := Nat.cast_nonneg _
  have he : (-5 - (head % H : Nat) : ℝ) * Real.log 2 < 0 :=
    mul_neg_of_neg_of_pos (by linarith) hlog
  exact sub_pos.mpr (by simpa [exp2] using Real.exp_lt_one_iff.mpr he)

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.Decay
