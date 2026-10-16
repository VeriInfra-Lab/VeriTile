import bench.tritonbench_g.int8_matmul_quantization.Proofs.QuantizeFirstPass
import bench.tritonbench_g.int8_matmul_quantization.Proofs.QuantizeWrites
namespace VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPQuantize
open VeriTile.Triton FP.Structural FP.IntegerConversion
open KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false
local instance secondWordDefault : Inhabited Word := ⟨FP.Value.zero .fp32⟩

structure SecondInvariant (cfg : FP.Config) (p : Parameters) (initial : State Word)
    (k : Nat) (current : State Word) : Prop where
  memory : current.mem = (written cfg p initial k).mem
  pidm : current.regs .nat [] "pid_m" = some (fun _ => initial.pids 0)
  offsk : current.regs .nat [p.blockColumns] "offs_k" = some (fun i => i.1.val)
  offsam : current.regs .nat [p.blockRows] "offs_am" = some (fun i => row p initial i.1)
  fptrs : current.regs .ptr [p.blockRows,p.blockColumns] "fpa_ptrs" =
    some (fun i => (p.input, inputOffset p initial k i))
  aptrs : current.regs .ptr [p.blockRows,p.blockColumns] "a_ptrs" =
    some (fun i => (p.output, outputOffset p initial k i))
  scaleval : current.regs .real [p.blockRows] "a_scale" = some (fun i => scale cfg p initial i.1)

def middleExecution (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (initial current : State Word) (h : FirstInvariant p initial p.steps current) :
    {t : State Word // run (backend rest cfg u) (middle p) current = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [middle, source, ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    bop, natLt, numericLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType, Region.cast, Option.bind,
    integralFloorDiv, integralMod, backend, QuantizeKvTransform.MixedBits.backend,
    QuantizeKvCopy.TypedQuantize.backend, fp32TotalBackend, fp32Source,
    FP.Software.scalarFP32, FP.Software.fp32Context,
    h.maxval, h.offsam, h.offsk]
  rfl

theorem middleInvariant (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (initial current : State Word) (h : FirstInvariant p initial p.steps current) :
    SecondInvariant cfg p initial 0 (middleExecution rest cfg u p initial current h).val := by
  constructor <;>
    simp [middleExecution, h.memory, h.pidm, h.offsk, h.offsam, h.aptrs,
      inputOffset, outputOffset, scale, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      show (127.0 : ℝ)=127 by norm_num]
  all_goals funext i
  all_goals simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]

def secondExecution (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (initial current : State Word) (k : Nat)
    (hsep : p.input ≠ p.output) (h : SecondInvariant cfg p initial k current) :
    {t : State Word // run (backend rest cfg u) (secondBody p)
      (current.setReg "k" .nat [] (fun _ => k)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [secondBody, source, ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    bop, natLt, numericLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType, Region.cast, Option.bind,
    integralFloorDiv, integralMod, backend, QuantizeKvTransform.MixedBits.backend,
    QuantizeKvCopy.TypedQuantize.backend, fp32TotalBackend, fp32Source,
    FP.Software.scalarFP32, FP.Software.fp32Context,
    h.offsk, h.fptrs, h.aptrs, h.scaleval, h.memory, written_input, hsep]
  rfl

private theorem conversion_defined (u : Word → Int) (x : Word) (h : (toSigned 8 x).isSome) :
    totalSigned u 8 x = (toSigned 8 x).getD 0 := by
  cases hx : toSigned 8 x with
  | none => simp [hx] at h
  | some z => simp [totalSigned, hx]

attribute [local irreducible] FP.div toSigned totalSigned

theorem secondInvariant (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (initial current : State Word) (k : Nat) (hk : k<p.steps)
    (hsep : p.input ≠ p.output) (hd : InputDefined cfg p initial)
    (h : SecondInvariant cfg p initial k current) :
    SecondInvariant cfg p initial (k+1) (secondExecution rest cfg u p initial current k hsep h).val := by
  constructor
  · simp +contextual [secondExecution, written_succ, masked_scatter_memory, h.memory,
      active, inputOffset, outputOffset, expected, loaded, quotient,
      FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      show (0.0 : ℝ)=0 by norm_num]
    apply congrArg (fun f => List.foldl f (written cfg p initial k).mem (TileShape.allIndices [p.blockRows,p.blockColumns]))
    funext mem i
    by_cases hi : active p k i
    · have hc := conversion_defined (u 8) _ (hd k hk i hi)
      simp only [active] at hi
      simp [quotient, loaded, active, inputOffset, hi] at hc
      simp only [hi, ↓reduceIte]
      funext r o
      change (if r=p.output ∧ o=outputOffset p initial k i then _ else mem r o) =
        (if r=p.output ∧ o=outputOffset p initial k i then _ else mem r o)
      split_ifs <;> try rfl
      apply congrArg (Cell.mk .int)
      exact hc
    · simp only [active] at hi
      simp only [hi, ↓reduceIte]
  all_goals
    simp [secondExecution, h.pidm, h.offsk, h.offsam, h.scaleval,
      inputOffset, outputOffset, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Nat.add_mul, Nat.add_assoc]
  all_goals funext i
  all_goals first
  | change (p.input, (row p initial i.1 * p.inputRowStride +
      (i.2.1.val*p.inputColumnStride + k*(p.blockColumns*p.inputColumnStride))) +
        p.blockColumns*p.inputColumnStride) = _
    simp [inputOffset, Nat.add_mul, Nat.add_assoc]
  | change (p.output, (row p initial i.1 * p.outputRowStride +
      (i.2.1.val*p.outputColumnStride + k*(p.blockColumns*p.outputColumnStride))) +
        p.blockColumns*p.outputColumnStride) = _
    simp [outputOffset, Nat.add_mul, Nat.add_assoc]

#axiomsClean middleExecution
#axiomsClean middleInvariant
#axiomsClean secondExecution
#axiomsClean secondInvariant
end
end VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPQuantize
