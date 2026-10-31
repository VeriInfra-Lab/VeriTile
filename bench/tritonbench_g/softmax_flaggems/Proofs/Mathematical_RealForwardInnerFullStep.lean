import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardInner
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealMath
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealLoop
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.KernelLemmas.ScatterStore

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

noncomputable def fullScore (s : BlockState) (X : RegionName)
    (row N start B : Nat) (i : TileIndex [B]) : WithBot ℝ :=
  some (s.readMem X (row * N + (start + i.1.val)))

private theorem option_max_eq (a b : WithBot ℝ) :
    @max (Option ℝ) Option.instMax a b = a ⊔ b := by
  cases a <;> cases b <;> rfl

private theorem exp_finite (a : WithBot ℝ) :
    WithBot.realExp a = some ((WithBot.realExp a).unbotD 0) := by
  cases a <;> rfl

private theorem update_z (m x : WithBot ℝ) (z : ℝ) :
    WithBot.realAdd
      (WithBot.realMul (some z) (WithBot.realExp (WithBot.realSub m (m ⊔ x))))
      (WithBot.realExp (WithBot.realSub x (m ⊔ x))) =
      some (RealMath.maskedOnlineStep (m, z) x).2 := by
  rw [exp_finite (WithBot.realSub m (m ⊔ x)),
    exp_finite (WithBot.realSub x (m ⊔ x))]
  rfl

theorem fullScore_eq_score (s : BlockState) (X : RegionName) (row N start B : Nat)
    (h : start + B ≤ N) (i : TileIndex [B]) :
    fullScore s X row N start B i = score s X row N start B i := by
  have hi := i.1.isLt
  simp [fullScore, score, show start + i.1.val < N by omega]

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
