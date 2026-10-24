import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.NativeOperands
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Native
open VeriTile Triton FP.Structural Source Whole
open VeriTile.Bench.Repairs.AttentionKernelAligned.Full
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000

/-- Exact launch-word staging, with no Real-to-bit recovery and no
assumption about a tensor or final result. Signed zero is retained. -/
theorem scale_word (p : Parameters) (u : Fin 4) :
    lower (scaleValue (modelFor p) p u)=
      .mul32 (.bits p.scaleWord) (.literal32 1.44269504) := by
  norm_num [scaleValue,modelFor,model,binaryFor,algebra,literal,lower]

theorem query_scale_word (p : Parameters) (s : State Expr) (i : Source.QLane p) (u : Fin 4) :
    lower (qScaled (modelFor p) p s i u)=
      .narrow16 (.mul32 (.mul32 (.bits p.scaleWord) (.literal32 1.44269504))
        (lower (widen (qValue (modelFor p) p s i u)))) := by
  norm_num [qScaled,scaleValue,modelFor,model,binaryFor,algebra,literal,narrow,lower]
  simp only [Option.some_ne_none,if_false,lower]

#axiomsClean scale_word
#axiomsClean query_scale_word
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Native
