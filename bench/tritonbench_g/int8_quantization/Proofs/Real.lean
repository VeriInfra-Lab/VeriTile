import bench.tritonbench_g.int8_quantization.Kernels
import VeriTile.Triton.Semantics.Scalar
import VeriTile.Meta.StatementAudit
import Mathlib.Data.Real.Sqrt

namespace VeriTile.Bench.Repairs.Int8Quantization.RealSpec
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000

/-- Independent symmetric nearest integer, with half ties away from zero. -/
def nearestAway (x : ℝ) : Int :=
  if 0≤ x then ⌊x+1/2⌋ else ⌈x-1/2⌉

/-- Independent logical block maximum; the empty block's scale is zero. -/
def maxAbs : List ℝ → ℝ
  | [] => 0
  | x::xs => max |x| (maxAbs xs)

def factor (query : Bool) (channels : Nat) : ℝ :=
  if query then (Real.sqrt (channels : ℝ))⁻¹ * (144269504/100000000) else 1

def target (query : Bool) (channels : Nat) (logicalBlock : List ℝ) : List Int × ℝ :=
  let xs := logicalBlock.map (fun x=> factor query channels*x)
  let m := maxAbs xs
  (xs.map (fun x=> nearestAway (127*x/m)),m/127)

theorem maxAbs_nonneg (xs : List ℝ) : 0≤ maxAbs xs := by
  cases xs with
  | nil=> rfl
  | cons x xs=> exact le_trans (abs_nonneg x) (le_max_left _ _)

theorem abs_le_maxAbs (xs : List ℝ) (x : ℝ) (member : x∈xs) : |x|≤ maxAbs xs := by
  induction xs with
  | nil=> simp at member
  | cons y ys ih=>
    rcases List.mem_cons.mp member with equal|tail
    · subst x;exact le_max_left _ _
    · exact le_trans (ih tail) (le_max_right _ _)

def clip (x : ℝ) : ℝ := min (max x (-127)) 127

/-- Real interpretation of the actual source's safe divisor, zero branch,
clamp, sign bias, and toward-zero cast, in precisely that order. -/
def sourceScalar (x scale : ℝ) : Int :=
  let denominator := if 0< scale then scale else 1
  let normalized := if 0< scale then x/denominator else 0
  let normalized := clip normalized
  tritonTruncTowardZero (normalized+(1/2)*(if 0≤ normalized then 1 else -1))

theorem bias_trunc (x : ℝ) :
    tritonTruncTowardZero (x+(1/2)*(if 0≤ x then 1 else -1))=nearestAway x := by
  by_cases sign : 0≤ x
  · simp only [if_pos sign,mul_one,nearestAway,tritonTruncTowardZero]
    rw [if_pos (by linarith : 0≤ x+1/2)]
  · simp only [if_neg sign,mul_neg_one,nearestAway,tritonTruncTowardZero]
    rw [if_neg (by linarith : ¬0≤ x+ -(1/2))]
    rw [Int.floor_neg]
    simp [sub_eq_add_neg]

theorem nearest_range (x : ℝ) (lo : -127≤ x) (hi : x≤127) :
    -127≤ nearestAway x ∧ nearestAway x≤127 := by
  by_cases sign : 0≤ x
  · rw [nearestAway,if_pos sign]
    have low : (0:Int)≤⌊x+1/2⌋ := Int.le_floor.mpr (by linarith)
    have high : ⌊x+1/2⌋<(128:Int) := Int.floor_lt.mpr (by norm_num;linarith)
    omega
  · rw [nearestAway,if_neg sign]
    have low : (-128:Int)<⌈x-1/2⌉ := Int.lt_ceil.mpr (by norm_num;linarith)
    have high : ⌈x-1/2⌉≤(0:Int) := Int.ceil_le.mpr (by norm_num;linarith)
    omega

theorem source_int8_range (x scale : ℝ) :
    -127≤ sourceScalar x scale ∧ sourceScalar x scale≤127 := by
  unfold sourceScalar
  rw [bias_trunc]
  apply nearest_range
  · exact le_min (le_max_right _ _) (by norm_num)
  · exact min_le_right _ _

theorem source_byte_exact (x scale : ℝ) :
    (BitVec.ofInt 8 (sourceScalar x scale)).toInt=sourceScalar x scale := by
  have range:=source_int8_range x scale
  apply BitVec.toInt_ofInt_eq_self (by decide)
  · norm_num;omega
  · norm_num;omega

theorem normalized_bounds (x m : ℝ) (hm : 0< m) (hx : |x|≤ m) :
    -127≤ x/(m/127) ∧ x/(m/127)≤127 := by
  have hs : 0< m/127 := by positivity
  rcases abs_le.mp hx with ⟨lo,hi⟩
  constructor
  · apply (le_div_iff₀ hs).mpr
    nlinarith
  · apply (div_le_iff₀ hs).mpr
    nlinarith

/-- The clamp is inert on every mathematically valid active lane. The zero
case is handled explicitly, rather than assumed away by a positive-scale law. -/
theorem scalar_correct (x m : ℝ) (hm : 0≤ m) (hx : |x|≤ m) :
    sourceScalar x (m/127)=nearestAway (127*x/m) := by
  by_cases zero : m=0
  · subst m
    simp [sourceScalar,clip,nearestAway,tritonTruncTowardZero]
  · have positive : 0< m := lt_of_le_of_ne hm (Ne.symm zero)
    have scale : 0< m/127 := by positivity
    obtain ⟨lo,hi⟩:=normalized_bounds x m positive hx
    have unclip : clip (x/(m/127))=x/(m/127) := by
      simp [clip,max_eq_left lo,min_eq_left hi]
    have ratio : x/(m/127)=127*x/m := by field_simp
    unfold sourceScalar
    simp only [if_pos scale,unclip]
    rw [bias_trunc,ratio]

def blockProgram (query : Bool) (channels : Nat) (logicalBlock : List ℝ) : List Int × ℝ :=
  let xs:=logicalBlock.map (fun x=> factor query channels*x)
  let scale:=maxAbs xs/127
  (xs.map (fun x=> sourceScalar x scale),scale)

/-- Complete mathematical block formula, including the scale producer and
stored signed integer values. Connecting this to typed source registers and
the host's physical calls is a separate program obligation. -/
theorem block_correct (query : Bool) (channels : Nat) (logicalBlock : List ℝ) :
    blockProgram query channels logicalBlock=target query channels logicalBlock := by
  apply Prod.ext
  · simp only [blockProgram,target]
    apply List.map_congr_left
    intro x hx
    exact scalar_correct x _ (maxAbs_nonneg _) (abs_le_maxAbs _ x hx)
  · rfl

#axiomsClean bias_trunc
#axiomsClean source_byte_exact
#axiomsClean scalar_correct
#axiomsClean block_correct
end
end VeriTile.Bench.Repairs.Int8Quantization.RealSpec
