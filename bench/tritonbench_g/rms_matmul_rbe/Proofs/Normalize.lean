import bench.tritonbench_g.rms_matmul_rbe.Proofs.Arithmetic

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

def normalization (p : Arguments) := (epilogue Bool.false Bool.false p).take 3

theorem normalization_source (fp8 outputFP8 : Bool) (p : Arguments) :
    (epilogue fp8 outputFP8 p).take 3=normalization p := by
  cases fp8 <;> cases outputFP8 <;> rfl

def rowNorm (a : Arithmetic α) (p : Arguments) (epsilon : α)
    (moments : Values α .real [16,64]) (row : Fin 16) : α :=
  a.rsqrt (a.add (a.divide (reduceLine a (fun k => moments (row,k,PUnit.unit)))
    (a.ofInt p.K)) epsilon)

def normalized (a : Arithmetic α) (p : Arguments) (epsilon : α)
    (moments accumulator : Values α .real [16,64]) : Values α .real [16,64] := fun i =>
  a.mul (accumulator i) (rowNorm a p epsilon moments i.1)

def normalizationExecution {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Arguments) (s : State α) (epsilon : α)
    (moments accumulator : Values α .real [16,64])
    (he : s.regs .real [] "EPS"=some (fun _ => epsilon))
    (hm : s.regs .real [16,64] "x_sum"=some moments)
    (ha : s.regs .real [16,64] "accumulator"=some accumulator) :
    {next : State α // run a.model (normalization p) s=some next} := by
  refine ⟨?_,?_⟩
  swap
  simp [normalization,epilogue,source,TypedSource.kernel,TypedSource.halfHalf,TypedSource.halfFP8,TypedSource.fp8Half,TypedSource.fp8FP8,ComputeKernel.surfaceBody,run,
    FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,
    numeric,bop,natLt,numericLt,ComputeDType.eraseDType,ofFloat,toFloat,
    Region.cast,Option.bind,integralFloorDiv,integralMod,Arithmetic.model,he,hm,ha,
    Broadcast.leftIndex,Broadcast.rightIndex]
  rfl

theorem normalization_value {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Arguments) (s : State α) (epsilon : α)
    (moments accumulator : Values α .real [16,64])
    (he : s.regs .real [] "EPS"=some (fun _ => epsilon))
    (hm : s.regs .real [16,64] "x_sum"=some moments)
    (ha : s.regs .real [16,64] "accumulator"=some accumulator) :
    (normalizationExecution a p s epsilon moments accumulator he hm ha).val.regs
      .real [16,64] "accumulator"=some (normalized a p epsilon moments accumulator) := by
  simp [normalizationExecution,normalized,rowNorm,Arithmetic.model,binary,
    FP.Structural.bop,reduceAxis,TileShape.insertAxisIndex,
    Broadcast.leftIndex,Broadcast.rightIndex,ofFloat,toFloat]
  rfl

theorem normalization_frame {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Arguments) (s : State α) (epsilon : α)
    (moments accumulator : Values α .real [16,64])
    (he : s.regs .real [] "EPS"=some (fun _ => epsilon))
    (hm : s.regs .real [16,64] "x_sum"=some moments)
    (ha : s.regs .real [16,64] "accumulator"=some accumulator) :
    (normalizationExecution a p s epsilon moments accumulator he hm ha).val.mem=s.mem := rfl

#axiomsClean normalization_source
#axiomsClean normalization_value
#axiomsClean normalization_frame
#specNonCircular normalized avoiding [normalizationExecution,run]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
