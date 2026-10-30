import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelTarget
import bench.tritonbench_g.rms_rbe_matmul.Kernels
import Mathlib.Analysis.Real.Pi.Bounds

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelCounterexample
open VeriTile.Triton
noncomputable section
namespace Original
abbrev source := VeriTile.Bench.TritonBenchG.RmsRbeMatmul.TypedSource.kernel
end Original

/-- The original matmul source ignores all three rotary arguments. This is
a source identity, not an equivalence claim about the canonical repair. -/
theorem original_ignores_rotary (M N K : Nat) (start other : Int) (theta : ℝ) :
    Original.source .fp16 Bool.false Bool.false "X" "W" "RMS" "OUT"
      M N K (M*K) K 1 1 K 1 (M*N) N 1 16 64 64 start Bool.true theta (1/1000000) =
    Original.source .fp16 Bool.false Bool.false "X" "W" "RMS" "OUT"
      M N K (M*K) K 1 1 K 1 (M*N) N 1 16 64 64 other Bool.false 10000 (1/1000000) := rfl

def pairInput : KernelMath.Input :=
  ⟨fun _ => 1, fun _ => 1, fun column _ => if column = 0 then 1 else 0⟩

/-- Safe full K64 block, one row, two output columns: the unrotated second
column is zero, while the intended rotation at integer position one is
strictly positive. This is independent of any tail-access failure. -/
theorem missing_rotary_changes_output :
    KernelMath.output 64 2 1 Bool.false pairInput 0 1 = 0 ∧
    0 < KernelMath.output 64 2 1 Bool.true pairInput 0 1 := by
  constructor
  · simp [KernelMath.output, KernelMath.projection, pairInput]
  · have hs : 0 < Real.sin 1 :=
      Real.sin_pos_of_pos_of_lt_pi (by norm_num) (by linarith [Real.pi_gt_three])
    have hi := KernelMath.inverse_positive 64 pairInput
    have h : 0 < (64 * KernelMath.inverse 64 pairInput) * Real.sin 1 := by positivity
    simpa [KernelMath.output, KernelMath.angle, KernelMath.projection,
      pairInput, KernelMath.theta] using h

/-- An exactly sized X[1,1,1] has no element at lane one of the original
unmasked K64 load. The repair's first-block mask excludes that lane. -/
theorem original_tail_and_repaired_mask :
    (1 : Nat) < 64 ∧ ¬(1 < 1) ∧ ¬(1 < 1 - 0*64) := by decide

#axiomsClean original_ignores_rotary
#axiomsClean missing_rotary_changes_output
#axiomsClean original_tail_and_repaired_mask
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelCounterexample
