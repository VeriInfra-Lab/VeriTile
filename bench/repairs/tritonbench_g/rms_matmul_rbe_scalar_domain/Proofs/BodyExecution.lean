import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.Arithmetic

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

def Int8Memory {α : Type} [Inhabited α] (s : State α) : Prop :=
  ∀ region offset,Float8E5Bits.isInt8 ((s.mem region offset).read .int)=Bool.true

theorem masked_int8 {α : Type} [Inhabited α] (s : State α) (bytes : Int8Memory s)
    (r : RegionName) (o : Nat) (mask : Prop) [Decidable mask] :
    Float8E5Bits.isInt8 (if mask then (s.mem r o).read .int else 0)=Bool.true := by
  split_ifs
  · exact bytes r o
  · decide

structure LoopValues (α : Type) where
  keys : Values α .nat [64]
  xptrs : Values α .ptr [16,64]
  wptrs : Values α .ptr [64,64]
  rmsptrs : Values α .ptr [1,64]
  accumulator : Values α .real [16,64]
  moments : Values α .real [16,64]

structure LoopRegisters (v : LoopValues α) (s : State α) : Prop where
  keys : s.regs .nat [64] "offs_k"=some v.keys
  xptrs : s.regs .ptr [16,64] "x_ptrs"=some v.xptrs
  wptrs : s.regs .ptr [64,64] "w_ptrs"=some v.wptrs
  rmsptrs : s.regs .ptr [1,64] "rms_w_ptrs"=some v.rmsptrs
  accumulator : s.regs .real [16,64] "accumulator"=some v.accumulator
  moments : s.regs .real [16,64] "x_sum"=some v.moments

/-- Construct the next state of the complete K64 source body, including
integer power, optional E5 bitcasts, widening/narrowing and pointer advances. -/
def bodyExecution {α : Type} [Inhabited α] (a : Arithmetic α)
    (fp8 outputFP8 : Bool) (p : Arguments) (s : State α)
    (v : LoopValues α) (h : LoopRegisters v s) (bytes : Int8Memory s) (t : Nat) :
    {next : State α // run a.model (body fp8 outputFP8 p)
      (s.setReg "step" .nat [] (fun _ => t))=some next} := by
  cases fp8 <;> cases outputFP8
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [body,source,TypedSource.kernel,ComputeKernel.surfaceBody,run,
      FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,
      numeric,bop,natLt,numericLt,ComputeDType.eraseDType,ofFloat,toFloat,
      Region.cast,Option.bind,integralFloorDiv,integralMod,Arithmetic.model,
      h.keys,h.xptrs,h.wptrs,h.rmsptrs,h.accumulator,h.moments,
      Broadcast.leftIndex,Broadcast.rightIndex,List.all_eq_true,masked_int8 s bytes]
    rfl

#axiomsClean masked_int8
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
