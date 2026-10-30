import bench.tritonbench_g.rms_matmul_rbe.Proofs.IOByteSourceIteration
namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOBytes
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
def nativeIteration (a : Arithmetic α) (fp8 outputFP8 : Bool) (p : Arguments) (s : State α)
    (v : LoopValues α) (h : LoopRegisters v s) (bytes : InputBytes fp8 p s)
    (regions : PointerRegions p v) (t : Nat) :
    {next : State α // run a.model (selectedBody fp8 outputFP8 p)
      (s.setReg "step" .nat [] (fun _=>t))=some next} := by
  have hw:=masked_weight fp8 p s bytes v regions
  have hr:=masked_rms fp8 p s bytes v regions
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
      Broadcast.leftIndex,Broadcast.rightIndex,List.all_eq_true,hw,hr]
    rfl


#axiomsClean nativeIteration
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOBytes
