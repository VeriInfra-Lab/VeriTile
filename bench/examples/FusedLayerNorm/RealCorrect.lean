import bench.examples.FusedLayerNorm.IO
import bench.examples.FusedLayerNorm.Kernels
import bench.examples.FusedLayerNorm.Proofs.Real
import bench.examples.Welford.Proofs.Real
import VeriTile.Meta.StatementAudit

/-! Real correctness of the sources in Kernels.lean.
IO contracts and mathematical formulas are stated below; execution and memory
lemmas are in Proofs/Real.lean. Numerical casts are interpreted over the reals. -/

namespace VeriTile.Bench.Examples.FusedLayerNormCorrect
open VeriTile.Bench.Examples.FusedLayerNorm.Kernels
open VeriTile Triton
open VeriTile.Triton.TiledReduction.WelfordRec
open VeriTile.Examples (InputRowLoadedAt onlineWelfordLoopBody)
open WelfordCorrect (stepStmt_assign_inv welfordLoop_traceSafe welfordBody_pid_preserved)
open scoped VeriTile.Triton.KernelIO₃

specification two_pass_layernorm_correct (N stride : Nat) (hWord : N < 2^64) (ε : ℝ) :
    Spec.Real (twoPassIO N stride ε ⊨ fun xs gs bs i =>
      let μ := (∑ j, xs j) / N
      let variance := (∑ j, (xs j - μ) ^ 2) / N
      (xs i - μ) / Real.sqrt (variance + ε) * gs i + bs i) := by
  have halg : (twoPassIO N stride ε).kernel.toAlgKernel =
      (twoPassMathKernel "x" "gamma" "beta" "y" N stride ε).toAlgKernel := by
    rfl
  refine KernelIO₃.Implements.intro _ ?_ ?_ ?_
  · rw [halg]; exact twoPass_flattenOk "x" "gamma" "beta" "y" N stride ε
  · intro bounds s hx hg hb hy _
    rw [halg]
    simp only [twoPassIO, Nat.zero_add] at hg hb
    exact twoPass_traceSafe "x" "gamma" "beta" "y" N stride ε s bounds hx hg hb hy
  · intro s xs gs bs hx hg hb
    rw [halg]
    simp only [twoPassIO, Nat.zero_add] at hg hb
    obtain ⟨t, he, hv, hf⟩ := twoPass_region_run N stride hWord ε s xs gs bs hx hg hb
    refine ⟨t, he, ?_, ?_⟩
    · simpa [twoPassMean, twoPassS] using hv
    · intro r o hmiss _; exact hf r o hmiss

specification fused_layernorm_correct (N stride : Nat) (hWord : N < 2^64) (ε : ℝ) :
    Spec.Real (fusedIO N stride ε ⊨ fun xs gs bs i =>
      let μ := (∑ j, xs j) / N
      let variance := (∑ j, (xs j - μ) ^ 2) / N
      (xs i - μ) / Real.sqrt (variance + ε) * gs i + bs i) := by
  have halg : (fusedIO N stride ε).kernel.toAlgKernel =
      (fusedMathKernel "x" "gamma" "beta" "y" N stride ε).toAlgKernel := by
    rfl
  refine KernelIO₃.Implements.intro _ ?_ ?_ ?_
  · rw [halg]; exact fused_flattenOk "x" "gamma" "beta" "y" N stride ε
  · intro bounds s hx hg hb hy _
    rw [halg]
    simp only [fusedIO, Nat.zero_add] at hg hb
    exact fused_traceSafe "x" "gamma" "beta" "y" N stride ε s bounds hx hg hb hy
  · intro s xs gs bs hx hg hb
    rw [halg]
    simp only [fusedIO, Nat.zero_add] at hg hb
    obtain ⟨t, he, hv, hf⟩ := fused_region_run N stride hWord ε s xs gs bs hx hg hb
    refine ⟨t, he, ?_, ?_⟩
    · simpa [twoPassMean, twoPassS] using hv
    · intro r o hmiss _; exact hf r o hmiss

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.FusedLayerNormCorrect
