import bench.tritonbench_g.rms_matmul_rbe.Proofs.Source

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 12000000
set_option linter.unusedSimpArgs false

/-- Successful execution of the actual shared-quotation prologue. Its state
is constructed from the caller state; success is not an ABI premise. -/
def prefixExecution {α : Type} [Inhabited α] (M : Algebra α)
    (fp8 outputFP8 : Bool) (p : Arguments) (s : State α) :
    {t : State α // run M (prelude fp8 outputFP8 p) s=some t} := by
  cases fp8 <;> cases outputFP8
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [prelude,source,TypedSource.kernel,TypedSource.halfHalf,TypedSource.halfFP8,TypedSource.fp8Half,TypedSource.fp8FP8,ComputeKernel.surfaceBody,run,
      FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,
      numeric,bop,natLt,numericLt,ComputeDType.eraseDType,ofFloat,toFloat,
      Region.cast,Option.bind,integralFloorDiv,integralMod]
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
