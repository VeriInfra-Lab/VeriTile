import Mathlib
import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.quantize_global.Sources.Mathematical_Kernels

local macro "true" : term => `(Bool.true)
local macro "false" : term => `(Bool.false)

/-! Partial shared mathematical scalar for the two quantization wrappers.
This is not a runtime dtype, a native compiler proof, or a complete wrapper.
`none` is an undefined mathematical value, not a host exception or rollback.
The eventual wrapper must continue ordinary host/setup stages in source order.
-/
namespace QuantizationReal
noncomputable section
open Classical
set_option maxHeartbeats 2000000

/-- Exact nearest integer, with ties to the even integer. Using floor retains
the correct negative-half behavior rather than silently rounding toward zero. -/
def nearestEven (x : ℝ) : Int :=
  let lower := Int.floor x
  let remainder := x - lower
  if (1 / 2 : ℝ) < remainder ∨ (remainder = 1 / 2 ∧ lower % 2 = 1)
  then lower + 1 else lower

def int64Representable (z : Int) : Prop := -(2^63 : Int) ≤ z ∧ z < 2^63

/-- Round first, check the signed64 domain, then retain the actual i64 word. -/
def llrint64 (x : ℝ) : Option (BitVec 64) :=
  let z := nearestEven x
  if int64Representable z then some (BitVec.ofInt 64 z) else none

/-- Raw signed64 to int8 narrowing is modular, never saturation/clipping. -/
def narrow8 (z : BitVec 64) : BitVec 8 := BitVec.ofInt 8 z.toInt

/-- Zero absmax has no finite Real reciprocal. This marker is not a new
exception at the reciprocal source stage. -/
def reciprocal (maximum : ℝ) : Option ℝ :=
  if maximum = 0 then none else some (1 / maximum)

inductive ScalarType where | real | integer64 | output8
abbrev Carrier : ScalarType → Type
  | .real => ℝ
  | .integer64 => BitVec 64
  | .output8 => BitVec 8

inductive Expr : ScalarType → Type where
  | input : Expr .real
  | inverse : Expr .real
  | literal127 : Expr .real
  | multiply : Expr .real → Expr .real → Expr .real
  | round64 : Expr .real → Expr .integer64
  | narrow : Expr .integer64 → Expr .output8

def Expr.eval (x : ℝ) (inverse : Option ℝ) : Expr t → Option (Carrier t)
  | .input => some x
  | .inverse => inverse
  | .literal127 => some 127
  | .multiply a b => do pure ((← a.eval x inverse) * (← b.eval x inverse))
  | .round64 a => (a.eval x inverse).bind llrint64
  | .narrow a => (a.eval x inverse).map narrow8

/-- The shared source scalar preserves its exact parenthesization and
llrint64-before-int8 sequence. Host and address structure are task-specific. -/
def source : Expr .output8 :=
  QuantizationSource.scalar .input .inverse .literal127 .multiply .round64 .narrow

/-- Independent formula on the actual kernel input and source-produced scale. -/
def result (x : ℝ) (inverse : Option ℝ) : Option (BitVec 8) :=
  inverse.bind fun v => (llrint64 (127 * (x * v))).map narrow8

theorem source_scalar (x : ℝ) (inverse : Option ℝ) :
    source.eval x inverse = result x inverse := by
  cases inverse <;> rfl

theorem reciprocal_zero : reciprocal 0 = none := by simp [reciprocal]

theorem reciprocal_nonzero (m : ℝ) (h : m ≠ 0) :
    reciprocal m = some (1 / m) := by simp [reciprocal,h]

theorem zero_normalizer_has_no_scalar_value (x : ℝ) :
    result x (reciprocal 0) = none := by simp [reciprocal_zero,result]

theorem llrint_defined_iff (x : ℝ) :
    (llrint64 x).isSome = true ↔ int64Representable (nearestEven x) := by
  by_cases h : int64Representable (nearestEven x) <;> simp [llrint64,h]

theorem half_ties :
    nearestEven (1/2) = 0 ∧ nearestEven (3/2) = 2 ∧ nearestEven (5/2) = 2 ∧
    nearestEven (-1/2) = 0 ∧ nearestEven (-3/2) = -2 ∧ nearestEven (-5/2) = -2 := by
  norm_num [nearestEven]

theorem nearestEven_integer (z : Int) : nearestEven (z : ℝ) = z := by
  simp [nearestEven]

theorem nearestEven_distance (x : ℝ) : |x - (nearestEven x : ℝ)| ≤ 1/2 := by
  have lower := Int.floor_le x
  have upper := Int.lt_floor_add_one x
  unfold nearestEven
  dsimp only
  split
  · rename_i roundedUp
    push_cast
    rw [abs_le]
    rcases roundedUp with strict | ⟨tie,_⟩ <;> constructor <;> linarith
  · rename_i roundedDown
    push Not at roundedDown
    rw [abs_le]
    constructor <;> linarith [roundedDown.1]

theorem nearestEven_tie_is_even (x : ℝ) (tie : x - (Int.floor x : ℝ) = 1/2) :
    nearestEven x % 2 = 0 := by
  unfold nearestEven
  dsimp only
  by_cases odd : Int.floor x % 2 = 1
  · simp [tie,odd,Int.add_emod]
  · have even : Int.floor x % 2 = 0 := by omega
    simp [tie,even]

theorem signed64_boundaries :
    llrint64 ((2^63 : Int) : ℝ) = none ∧
    llrint64 ((-(2^63) : Int) : ℝ) = some (BitVec.ofInt 64 (-(2^63))) := by
  norm_num [llrint64,nearestEven,int64Representable]

theorem narrowing_is_not_clipping :
    narrow8 (BitVec.ofInt 64 128) = BitVec.ofInt 8 (-128) ∧
    narrow8 (BitVec.ofInt 64 255) = BitVec.ofInt 8 (-1) := by decide

/-- A strided host view may have maximum 1 while the kernel's linear read
consumes an in-storage value 2^63 outside that view. Exact scaling then exceeds
llrint64; no unconditional [-127,127] quantization bound is available. -/
theorem linear_read_outside_logical_maximum :
    result ((2^63 : Int) : ℝ) (reciprocal 1) = none := by
  norm_num [result,reciprocal,llrint64,nearestEven,int64Representable]

#specNonCircular result avoiding [source,Expr.eval]
#specNonCircular int64Representable avoiding [source,Expr.eval,llrint64,result]
#axiomsClean source_scalar
#axiomsClean reciprocal_zero
#axiomsClean reciprocal_nonzero
#axiomsClean zero_normalizer_has_no_scalar_value
#axiomsClean llrint_defined_iff
#axiomsClean half_ties
#axiomsClean nearestEven_integer
#axiomsClean nearestEven_distance
#axiomsClean nearestEven_tie_is_even
#axiomsClean signed64_boundaries
#axiomsClean narrowing_is_not_clipping
#axiomsClean linear_read_outside_logical_maximum
#guard_msgs (drop info) in
#auditModuleAxioms
end
end QuantizationReal
