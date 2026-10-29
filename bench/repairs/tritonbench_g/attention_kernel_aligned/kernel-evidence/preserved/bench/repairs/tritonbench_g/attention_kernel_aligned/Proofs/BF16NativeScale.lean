import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.BF16NativeOperands
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Native
open VeriTile Triton FP.Structural Source Whole
open VeriTile.Bench.Repairs.AttentionKernelAligned.Full
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000

/-- Exact launch-word staging, with no Real-to-bit recovery and no
assumption about a tensor or final result. Signed zero is retained. -/
theorem scale_word (p : Parameters) (u : Fin 5) :
    lower (scaleValue (modelFor p) p u)=
      .mul32 (.bits p.scaleWord) (.literal32 1.44269504) := by
  norm_num [scaleValue,modelFor,model,binaryFor,algebra,literal,lower]

theorem query_scale_word (p : Parameters) (s : State Expr) (i : Source.QLane p) (u : Fin 5) :
    lower (qScaled (modelFor p) p s i u)=
      .narrowBF16 (.mul32 (.mul32 (.bits p.scaleWord) (.literal32 1.44269504))
        (lower (widen (qValue (modelFor p) p s i u)))) := by
  norm_num [qScaled,scaleValue,modelFor,model,binaryFor,algebra,literal,narrow,lower]
  have hm : Binary.mul ≠ Binary.div := by decide
  simp only [Option.some_ne_none,hm,false_and,if_false,lower]

#axiomsClean scale_word
#axiomsClean query_scale_word
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Native
