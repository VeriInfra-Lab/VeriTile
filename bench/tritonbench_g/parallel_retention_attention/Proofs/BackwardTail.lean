import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelBackwardWhole
namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPStages
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
  (BackwardConfig BackwardAccumulator DotPrimitive withDot backwardOutputMemory)
noncomputable section
set_option maxHeartbeats 500000

def backwardTail (c : BackwardConfig) := (backwardKernel c).surfaceBody.drop 28

theorem backward_tail_split (c : BackwardConfig) :
    backwardTail c=backwardNumericalPrefix c++[backwardFirstRange c]++backwardMiddle c++
      [backwardSecondRange c]++backwardSuffix c:=rfl

theorem backward_tail_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (s captured : State α) (capturedArgs : BackwardArgumentReady c s captured) (bounds : BackwardIndexBounds c) (positive : 0<c.BTS) :
    ∃ final, run (withDot M D) (backwardTail c) captured = some final ∧
      final.mem = (backwardOutputMemory M c s (backwardValues M D c s)).mem := by
  obtain ⟨initial, hp, hi⟩ := backward_numerical_run M D c s captured capturedArgs bounds.toHelperIndexBounds
  obtain ⟨first, hr, hf⟩ := backward_first_range_run M D c s initial 0 (by omega) _ hi bounds
  simp only [Nat.cast_zero,zero_mul,add_zero] at hr
  have stopBound : (c.i_c+1)*c.BTL≤2147483648 := by
    have := bounds.time
    nlinarith
  obtain ⟨diagonal, hm, hd⟩ := backward_middle_run M D c s first _ hf
  obtain ⟨second, hs, hz⟩ := backward_second_range_run M D c s diagonal (c.i_c*c.BTL) ((c.i_c+1)*c.BTL) _ hd bounds.toHelperIndexBounds stopBound
  obtain ⟨final, he, hout⟩ := backward_suffix_run M D c s second _ hz bounds.toHelperIndexBounds
  refine ⟨final, ?_, ?_⟩
  · rw [backward_tail_split]
    simp only [run_append, hp, bind, pure, Option.bind,run,backward_first_range_statement M D c s initial _ hi positive,
      hr,hm,backward_second_range_statement M D c s diagonal _ hd.toBackwardReady,hs,he]
  · simpa only [backwardValues, backwardInitial] using hout


#axiomsClean backward_tail_run
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPStages
