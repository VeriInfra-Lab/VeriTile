import bench.tritonbench_g.parallel_attention.Proofs.FPShape
import bench.tritonbench_g.parallel_attention.Proofs.FPWitness
namespace VeriTile.Bench.TritonBenchG.ParallelAttention.FPCorrect
open VeriTile.Triton FP.Structural
open FusedRecurrentDelta.FPStages (sequentialFP32)
open FPStages
noncomputable section

/-- Both original forward loops, O and Z, and the exact memory frame. -/
specification forward_fp32 {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s : State α) :
    ∃ final, FP.Structural.exec (withDot M D) (forwardKernel c) s = some final ∧
      final.mem = (forwardMemory M D c s).mem := by
  exact FPStages.forward_fp32 M D c s

/-- Both original backward DK/DV loops and the exact memory frame. -/
specification backward_fp32 {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s : State α) :
    ∃ final, FP.Structural.exec (withDot M D) (backwardKernel c) s = some final ∧
      final.mem = (backwardOutputMemory M c s (backwardValues M D c s)).mem := by
  exact FPStages.backward_fp32 M D c s

/-- Full forward source instantiated with genuine fp32 scalar rounding,
ordered multiply/add dot and ordered sum. Scale=1 closes every used primitive
without depending on the supplied unused backend operations. -/
specification forward_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (c : ForwardConfig)
    (s : State (FP.Value .fp32)) (hscale : c.scale=1) :
    ∃ final, FP.Structural.exec (withDot (sequentialFP32 rest cfg) (softwareDot cfg)) (forwardKernel c) s = some final ∧
      final.mem = (roundedOutputMemory c s (roundedForwardValues cfg c s)).mem := by
  exact FPStages.forward_software_fp32 rest cfg c s hscale

/-- Nonempty two-loop interpreter witness: 1+2^-25 rounds back to 1.
This unit-tile witness is not a CUDA launch-validity claim. -/
specification forward_nonempty_rounding [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (withDot (sequentialFP32 rest .ieeeValue) (softwareDot .ieeeValue))
      (forwardKernel roundingConfig) roundingInitial = some final ∧
      final.mem "out" 1 = .mk .real one32 ∧
      final.mem "out" 0 = roundingInitial.mem "out" 0 ∧
      (1 + 1/33554432 : Rat) ≠ 1 := by
  exact FPStages.forward_nonempty_rounding rest

#axiomsClean forward_fp32
#axiomsClean backward_fp32
#axiomsClean forward_software_fp32
#axiomsClean forward_nonempty_rounding
end
end VeriTile.Bench.TritonBenchG.ParallelAttention.FPCorrect
