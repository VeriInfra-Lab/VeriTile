import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardNonInnerLoop

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
open VeriTile Triton FullSource
open scoped BigOperators
set_option maxHeartbeats 2500000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

noncomputable def columnMax {B C : Nat} (hB : 0 < B)
    (mf : TileIndex [B, C] → WithBot ℝ) (k : Fin C) : WithBot ℝ :=
  Finset.univ.sup' (⟨⟨0, hB⟩, Finset.mem_univ _⟩ : (Finset.univ : Finset (Fin B)).Nonempty)
    (fun i ↦ mf (i, k, PUnit.unit))

noncomputable def columnSum {B C : Nat} (mf : TileIndex [B, C] → WithBot ℝ)
    (zf : TileIndex [B, C] → ℝ) (mx : Fin C → WithBot ℝ) (k : Fin C) : ℝ :=
  ∑ i : Fin B, zf (i, k, PUnit.unit) *
    (WithBot.realExp (WithBot.realSub (mf (i, k, PUnit.unit)) (mx k))).unbotD 0

private theorem mul_exp_finite (z : ℝ) (a : WithBot ℝ) :
    WithBot.realMul (some z) (WithBot.realExp a) =
      some (z * (WithBot.realExp a).unbotD 0) := by
  cases a <;> rfl

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
