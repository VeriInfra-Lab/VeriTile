import bench.tritonbench_g.int8_matmul_quantization.Proofs.MatmulSource
namespace VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPMatmul
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
local instance matmulExecutionDefault : Inhabited Word := ⟨FP.Value.zero .fp32⟩
attribute [local simp 2000] FP.Structural.eval_intToSigned
private theorem narrow_zero (cfg : FP.Config) :
    QuantizeKvTransform.MixedBits.narrow cfg (FP.Value.zero .fp32) = FP.Value.zero .fp32 := by
  rcases cfg with ⟨⟨⟩, ⟨⟩, ⟨⟩⟩ <;> decide +kernel
private theorem min_lt_form (a b : Nat) : min a b = if a < b then a else b := by
  by_cases h : a < b
  · rw [if_pos h, min_eq_left (Nat.le_of_lt h)]
  · rw [if_neg h, min_eq_right (Nat.le_of_not_gt h)]
structure Invariant (p : Parameters) (initial : State Word) (k : Nat) (current : State Word) : Prop where
  memory : current.mem = initial.mem
  pidm : current.regs .nat [] "pid_m" = some (fun _ => rowBlock p initial)
  pidn : current.regs .nat [] "pid_n" = some (fun _ => columnBlock p initial)
  offsk : current.regs .nat [p.blockInner] "offs_k" = some (fun i => inputInner p initial i.1)
  aptrs : current.regs .ptr [p.blockRows,p.blockInner] "a_ptrs" =
    some (fun i => (p.input, aOffset p initial k i))
  bptrs : current.regs .ptr [p.blockInner,p.blockColumns] "b_ptrs" =
    some (fun i => (p.weight, bOffset p initial k i))
  ascale : current.regs .fp16 [p.blockRows] "a_scale" = some (fun i => aScale p initial i.1)
  bscale : current.regs .real [p.blockColumns] "b_scale" = some (fun i => bScale p initial i.1)
  acc : current.regs .int [p.blockRows,p.blockColumns] "accumulator" = some (accumulation p initial k)

def preludeExecution (rest : Algebra Word) (cfg : FP.Config) (p : Parameters) (s : State Word) :
    {t : State Word // run (backend rest cfg) (prelude p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [prelude, source, typed, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType, Region.cast, Option.bind,
    integralFloorDiv, integralMod, backend, QuantizeKvTransform.MixedBits.backend,
    QuantizeKvCopy.TypedQuantize.backend, FP.IntegerConversion.fp32TotalBackend,
    FP.IntegerConversion.fp32Source, FP.Software.scalarFP32, FP.Software.fp32Context]
  rfl

theorem preludeInvariant (rest : Algebra Word) (cfg : FP.Config) (p : Parameters) (s : State Word) :
    Invariant p s 0 (preludeExecution rest cfg p s).val := by
  constructor <;>
    simp [preludeExecution, backend, accumulation, blocksM, blocksN, firstRow,
      rowsInGroup, rowBlock, columnBlock, inputInner, inputColumn, inputRow,
      aOffset, bOffset, aScale, bScale, narrow_zero, min_lt_form, FP.Structural.bop,
      Broadcast.leftIndex, Broadcast.rightIndex, show (0.0 : ℝ)=0 by norm_num]
  all_goals funext i
  all_goals try simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]


def bodyExecution (rest : Algebra Word) (cfg : FP.Config) (p : Parameters)
    (initial current : State Word) (k : Nat) (h : Invariant p initial k current) :
    {t : State Word // run (backend rest cfg) (body p)
      (current.setReg "k" .nat [] (fun _ => k)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [body, source, typed, widths, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType, Region.cast, Option.bind,
    integralFloorDiv, integralMod, h.offsk, h.aptrs, h.bptrs, h.acc, h.memory,
    backend, IntegerExpr.sameShape]
  rfl

theorem bodyInvariant (rest : Algebra Word) (cfg : FP.Config) (p : Parameters)
    (initial current : State Word) (k : Nat) (h : Invariant p initial k current) :
    Invariant p initial (k+1) (bodyExecution rest cfg p initial current k h).val := by
  constructor <;>
    simp [bodyExecution, h.memory, h.pidm, h.pidn, h.offsk, h.aptrs, h.bptrs,
      h.ascale, h.bscale, h.acc, accumulation, aValue, bValue, wrap,
      aOffset, bOffset, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Nat.add_mul, Nat.add_assoc]
  all_goals funext i
  all_goals try simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    aOffset, bOffset, Nat.add_assoc]


  all_goals rcases i with ⟨r,c,u⟩
  all_goals cases u
  all_goals rw [show aValue p initial k = (fun i => Int.bmod
      (if inputInner p initial i.2.1 < p.inner-k*p.blockInner then
        (initial.mem p.input (inputRow p initial i.1*p.inputRowStride+
          (inputInner p initial i.2.1*p.inputInnerStride+k*(p.blockInner*p.inputInnerStride)))).read .int else 0) 256) from by
    funext j
    simp [aValue, aOffset, wrap, Nat.add_assoc]]
  all_goals rw [show bValue p initial k = (fun i => Int.bmod
      (if inputInner p initial i.1 < p.inner-k*p.blockInner then
        (initial.mem p.weight (inputInner p initial i.1*p.weightInnerStride+
          (inputColumn p initial i.2.1*p.weightColumnStride+k*(p.blockInner*p.weightInnerStride)))).read .int else 0) 256) from by
    funext j
    simp [bValue, bOffset, wrap, Nat.add_assoc]]

#axiomsClean preludeExecution
#axiomsClean preludeInvariant
#axiomsClean bodyExecution
#axiomsClean bodyInvariant
end
end VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPMatmul
