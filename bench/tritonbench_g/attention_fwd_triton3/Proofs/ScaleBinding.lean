import bench.tritonbench_g.attention_fwd_triton3.Kernels
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.ScaleBinding
open VeriTile.Triton
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
noncomputable section

/-- A unique quotation parameter marker, distinct from every floating literal
in these two GPU functions. It denotes the raw runtime FP32 ABI argument. -/
def marker : ℝ := -1
def template (c : Call) : Call := {c with scale:=marker}

/-- The complete actual quotation is independent of the Real parameter away
from this one assignment. Native interpretation binds that parameter slot to
the supplied 32-bit word, preserving negative zero and NaN payloads. -/
theorem only_scale_slot (c : Call) :
    (Kernels.source c).surfaceBody=
      (Kernels.source (template c)).surfaceBody.take 17 ++
      [.assign .real [] "qk_scale" (.alg (.const c.scale))] ++
      (Kernels.source (template c)).surfaceBody.drop 18 := rfl

theorem marker_distinct : marker≠0 ∧ marker≠1 ∧ marker≠1.4426950408889634 := by
  refine ⟨by norm_num [marker],by norm_num [marker],?_⟩
  have nonnegative : (0 : ℝ)≤1.4426950408889634 := by positivity
  unfold marker
  linarith

def bindLiteral (word : BitVec 32) (ordinary : ℝ → BitVec 32) (x : ℝ) : BitVec 32 :=
  if x=marker then word else ordinary x

theorem raw_scale (word : BitVec 32) (ordinary : ℝ → BitVec 32) :
    bindLiteral word ordinary marker=word := by simp [bindLiteral]

theorem negative_zero (ordinary : ℝ → BitVec 32) :
    bindLiteral 0x80000000 ordinary marker=0x80000000 := raw_scale _ _

theorem constants_unchanged (word : BitVec 32) (ordinary : ℝ → BitVec 32) :
    bindLiteral word ordinary 0=ordinary 0 ∧
    bindLiteral word ordinary 1=ordinary 1 ∧
    bindLiteral word ordinary 1.4426950408889634=ordinary 1.4426950408889634 := by
  simp only [bindLiteral,Ne.symm marker_distinct.1,Ne.symm marker_distinct.2.1,
    Ne.symm marker_distinct.2.2,if_false,and_self]

#axiomsClean only_scale_slot
#axiomsClean marker_distinct
#axiomsClean raw_scale
#axiomsClean constants_unchanged
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.ScaleBinding
