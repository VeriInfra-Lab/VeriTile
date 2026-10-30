import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelCompiler
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelScale
open VeriTile.Triton KernelProgram KernelCompiler KernelNativeCarrier
noncomputable section

def launchWord (platform : Platform) (binding : Binding platform kind s f) (arg : ScaleArgument) : FP.Value .fp32 :=
  ⟨platform.scale binding.object (hostScale platform s arg)⟩
def realScale (platform : Platform) (binding : Binding platform kind s f) (arg : ScaleArgument) : ℝ :=
  ((launchWord platform binding arg).toRat?.getD 0 : ℝ)
theorem finite_projection (platform : Platform) (binding : Binding platform kind s f) (arg : ScaleArgument)
    (admitted : ScaleAdmitted platform binding arg) :
    ∃ q : ℚ,(launchWord platform binding arg).toRat?=some q ∧ realScale platform binding arg=(q:ℝ) := by
  have finite : (launchWord platform binding arg).exponent≠255 := by
    simpa [launchWord,FP.Value.exponent,FP.Value.raw,FP.Format.fractionBits,finite32] using admitted.2
  have existsRat : ∃ q : ℚ,(launchWord platform binding arg).toRat?=some q := by
    simp [FP.Value.toRat?,FP.Value.decode,finite]
  obtain ⟨q,hq⟩ := existsRat
  exact ⟨q,hq,by simp [realScale,hq]⟩

#axiomsClean finite_projection
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelScale
