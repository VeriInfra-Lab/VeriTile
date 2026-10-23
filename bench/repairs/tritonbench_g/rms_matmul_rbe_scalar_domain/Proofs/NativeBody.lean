import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.BodyExecution
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.Lowering

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

/-- Execution of the selected local add(dot)-to-carry contraction. This
constructs a distinct execution and makes no floating equality claim with
the source dot that starts from zero. -/
def nativeBodyExecution {α : Type} [Inhabited α] (a : Arithmetic α)
    (fp8 outputFP8 : Bool) (p : Arguments) (s : State α)
    (v : LoopValues α) (h : LoopRegisters v s) (bytes : Int8Memory s) (t : Nat) :
    {next : State α // run a.model (selectedBody fp8 outputFP8 p)
      (s.setReg "step" .nat [] (fun _ => t))=some next} := by
  cases fp8 <;> cases outputFP8
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [selectedBody,dotIndex,carryDotNode,fp,matrixRef,
      body,source,TypedSource.kernel,TypedSource.halfHalf,TypedSource.halfFP8,TypedSource.fp8Half,TypedSource.fp8FP8,ComputeKernel.surfaceBody,run,
      FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,
      numeric,bop,natLt,numericLt,ComputeDType.eraseDType,ofFloat,toFloat,
      Region.cast,Option.bind,integralFloorDiv,integralMod,Arithmetic.model,
      h.keys,h.xptrs,h.wptrs,h.rmsptrs,h.accumulator,h.moments,
      Broadcast.leftIndex,Broadcast.rightIndex,List.all_eq_true,masked_int8 s bytes]
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
