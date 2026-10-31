import bench.tritonbench_g.adam_update_triton.Kernels
import VeriTile.Triton.Math.Optimizer
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.AdamUpdateTriton.Math
open VeriTile.Triton
noncomputable section

/-- Independent input-only mathematical target. The source's filename is Adam,
but its single momentum state and sign step implement Lion. -/
def target (params : Scalars ℝ) (output : Output) (p g m : ℝ) : ℝ :=
  match output with
  | .parameter => TiledOptimizer.lionParam p m g params.lr params.wd params.beta1
  | .momentum => TiledOptimizer.lionMomentum m g params.beta2

def sourceScalar (params : Scalars ℝ) (output : Output) (p g m : ℝ) : ℝ :=
  let decay := p * (1 - params.lr * params.wd)
  let diff := m - g
  let update := diff * params.beta1 + g
  match output with
  | .parameter => decay + (if 0 < update then -params.lr else params.lr) * (if update ≠ 0 then 1 else 0)
  | .momentum => diff * params.beta2 + g

theorem source_scalar_correct (params : Scalars ℝ) (output : Output) (p g m : ℝ) :
    sourceScalar params output p g m = target params output p g m := by
  cases output
  · dsimp only [sourceScalar, target]
    unfold TiledOptimizer.lionParam
    have direction : (m-g)*params.beta1+g = TiledOptimizer.lionUpdateDir m g params.beta1 := by
      unfold TiledOptimizer.lionUpdateDir
      ring
    rw [direction]
    have step := TiledOptimizer.lion_step_eq_branch params.lr (TiledOptimizer.lionUpdateDir m g params.beta1)
    by_cases zero : TiledOptimizer.lionUpdateDir m g params.beta1 = 0
    · simp [zero, Real.sign_zero]
      ring
    · rw [if_neg zero] at step
      simp only [zero, ne_eq, not_false_eq_true, if_true, mul_one]
      rw [← step]
      ring
  · dsimp only [sourceScalar, target]
    unfold TiledOptimizer.lionMomentum
    ring

#specNonCircular target avoiding [sourceScalar, source]
#axiomsClean source_scalar_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.Math
