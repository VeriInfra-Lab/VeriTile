import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.LoopComplete

/-! The byte conversion checks concern W/RMS operands only. Unrelated integer
cells are not an input condition of the logical kernel interface. -/
namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOBytes
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def InputBytes (fp8 : Bool) (p : Arguments) (s : State α) : Prop :=
  fp8=Bool.true → ∀ r, r=p.W ∨ r=p.RMS → ∀ off,
    Float8E5Bits.isInt8 ((s.mem r off).read .int)=Bool.true

structure PointerRegions (p : Arguments) (v : LoopValues α) : Prop where
  weight : ∀ i,(v.wptrs i).1=p.W
  rms : ∀ i,(v.rmsptrs i).1=p.RMS

theorem masked_weight (fp8 : Bool) (p : Arguments) (s : State α) (bytes : InputBytes fp8 p s)
    (v : LoopValues α) (regions : PointerRegions p v) (i : TileIndex [64,64])
    (mask : Prop) [Decidable mask] (on : fp8=Bool.true) :
    Float8E5Bits.isInt8 (if mask then (s.mem (v.wptrs i).1 (v.wptrs i).2).read .int else 0)=Bool.true := by
  split_ifs
  · exact bytes on _ (Or.inl (regions.weight i)) _
  · decide

theorem masked_rms (fp8 : Bool) (p : Arguments) (s : State α) (bytes : InputBytes fp8 p s)
    (v : LoopValues α) (regions : PointerRegions p v) (i : TileIndex [1,64])
    (mask : Prop) [Decidable mask] (on : fp8=Bool.true) :
    Float8E5Bits.isInt8 (if mask then (s.mem (v.rmsptrs i).1 (v.rmsptrs i).2).read .int else 0)=Bool.true := by
  split_ifs
  · exact bytes on _ (Or.inr (regions.rms i)) _
  · decide

def sourceIteration (a : Arithmetic α) (fp8 outputFP8 : Bool) (p : Arguments) (s : State α)
    (v : LoopValues α) (h : LoopRegisters v s) (bytes : InputBytes fp8 p s)
    (regions : PointerRegions p v) (t : Nat) :
    {next : State α // run a.model (body fp8 outputFP8 p)
      (s.setReg "step" .nat [] (fun _=>t))=some next} := by
  have hw:=masked_weight fp8 p s bytes v regions
  have hr:=masked_rms fp8 p s bytes v regions
  cases fp8 <;> cases outputFP8
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [
      body,source,TypedSource.kernel,TypedSource.halfHalf,TypedSource.halfFP8,TypedSource.fp8Half,TypedSource.fp8FP8,ComputeKernel.surfaceBody,run,
      FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,
      numeric,bop,natLt,numericLt,ComputeDType.eraseDType,ofFloat,toFloat,
      Region.cast,Option.bind,integralFloorDiv,integralMod,Arithmetic.model,
      h.keys,h.xptrs,h.wptrs,h.rmsptrs,h.accumulator,h.moments,
      Broadcast.leftIndex,Broadcast.rightIndex,List.all_eq_true,hw,hr]
    rfl


#axiomsClean sourceIteration
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOBytes
