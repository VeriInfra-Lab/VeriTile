import bench.tritonbench_g.rms_matmul_rbe.Proofs.IOSafetyBody

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOSafety
open VeriTile.Triton FP.Structural Safety
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem native_body (a : Arithmetic α) (fp8 outputFP8 : Bool) (p : Arguments) (s : State α)
    (v : LoopValues α) (h : LoopRegisters v s) (bytes : IOBytes.InputBytes fp8 p s)
    (regions : IOBytes.PointerRegions p v) (t : Nat) (capacity : RegionName → Nat)
    (bound : LoopBounds p v t capacity) :
    traceSafeList a.model capacity (selectedBody fp8 outputFP8 p) (s.setReg "step" .nat [] (fun _=>t)) := by
  have byteW:=IOBytes.masked_weight fp8 p s bytes v regions
  have byteR:=IOBytes.masked_rms fp8 p s bytes v regions
  have hx:=bound.x
  have hw:=bound.w
  have hr:=bound.rms
  simp only [liveKey] at hx hw hr
  cases fp8 <;> cases outputFP8
  all_goals
    simp [selectedBody,dotIndex,carryDotNode,fp,matrixRef,body,source,TypedSource.kernel,TypedSource.halfHalf,TypedSource.halfFP8,TypedSource.fp8Half,TypedSource.fp8FP8,ComputeKernel.surfaceBody,
      traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
      addressSafe_unfold,maskActive_unfold,
      FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,
      numeric,bop,natLt,numericLt,ComputeDType.eraseDType,ofFloat,toFloat,
      Region.cast,Option.bind,integralFloorDiv,integralMod,Arithmetic.model,
      h.keys,h.xptrs,h.wptrs,h.rmsptrs,h.accumulator,h.moments,
      Broadcast.leftIndex,Broadcast.rightIndex,List.all_eq_true,byteW,byteR,hx,hw,hr]
    exact ⟨fun m k unit hk=>hx (m,k,unit) hk,
      fun k unit hk=>hr (0,k,unit) hk,fun k n unit hk=>hw (k,n,unit) hk⟩

#axiomsClean native_body
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOSafety
