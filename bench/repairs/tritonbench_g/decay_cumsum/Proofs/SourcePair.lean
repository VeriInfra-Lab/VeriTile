import bench.repairs.tritonbench_g.decay_cumsum.PairIO
import bench.tritonbench_g.decay_cumsum.Proofs.MaskEquivalence

namespace VeriTile.Bench.Repairs.DecayCumsum.SourcePair
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
noncomputable section

set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

@[spec_rule] theorem signed_lt {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (s : FP.Structural.State α) (before : List ComputeStmt) (bc : Broadcast sa sb sh)
    (a a' : Op .int sa) (b b' : Op .int sb)
    (ha : OpEquivalent M s before a a') (hb : OpEquivalent M s before b b') :
    OpEquivalent M s before (.lt .int bc a b) (.lt .int bc a' b') := by
  intro t ht
  simp only [FP.Structural.evalOp]
  rw [ha t ht,hb t ht]

/-- The six complete pasted bodies differ only in the signed integer mask
product. The rewrite preserves every state, failure and floating operation. -/
theorem equivalent (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) :
    original kind h p scale ≡[assumptions] indexCommuted kind h p scale := by
  cases kind <;> equiv_decompose <;> fp_prove

#print_fp_assumptions equivalent
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.DecayCumsum.SourcePair
