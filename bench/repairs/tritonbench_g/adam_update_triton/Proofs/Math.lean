import bench.repairs.tritonbench_g.adam_update_triton.Kernels
import bench.tritonbench_g.adam_update_triton.Proofs.GridFrameReuse
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

/-- Reuse the checked original/example complete-grid value and exterior frame.
This is a chunk execution lemma, not yet a claim about the repaired wrapper. -/
theorem original_chunk (params : Scalars ℝ) (total chunk block : Nat)
    (selected : block = 128 ∨ block = 1024) (initial : BlockState) :
    ∃ final : BlockState,
      Nonempty (Kernel.GridLaunchedOrdinary
        (mathematicalSource params total chunk block).toAlgKernel
        (VeriTile.Bench.Examples.AdamUpdateGridLaunch.adamGrid ((chunk+block-1)/block)) initial final) ∧
      (∀ i, i<chunk → final.readMem "P" i=target params .parameter (initial.readMem "P" i)
          (initial.readMem "G" i) (initial.readMem "M" i) ∧
        final.readMem "M" i=target params .momentum (initial.readMem "P" i)
          (initial.readMem "G" i) (initial.readMem "M" i)) ∧
      (∀ (region : RegionName) offset, (region≠"P" ∧ region≠"M") ∨ chunk≤offset →
        final.mem region offset=initial.mem region offset) := by
  exact VeriTile.Bench.TritonBenchG.AdamUpdateTriton.GridReuse.selected_config_real_grid_with_frame
    "P" "G" "M" params.lr params.wd params.beta1 params.beta2 chunk block selected (by decide) initial

#specNonCircular target avoiding [sourceScalar, source, mathematicalSource]
#axiomsClean source_scalar_correct
#axiomsClean original_chunk
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.Math
