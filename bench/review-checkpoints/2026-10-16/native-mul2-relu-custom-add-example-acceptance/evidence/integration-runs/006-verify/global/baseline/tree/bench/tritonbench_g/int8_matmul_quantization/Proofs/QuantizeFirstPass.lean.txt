import bench.tritonbench_g.int8_matmul_quantization.Proofs.QuantizeSource
namespace VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPQuantize
open VeriTile.Triton FP.Structural FP.IntegerConversion
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false
local instance firstWordDefault : Inhabited Word := ⟨FP.Value.zero .fp32⟩

structure FirstInvariant (p : Parameters) (initial : State Word) (k : Nat) (current : State Word) : Prop where
  memory : current.mem = initial.mem
  pidm : current.regs .nat [] "pid_m" = some (fun _ => initial.pids 0)
  offsk : current.regs .nat [p.blockColumns] "offs_k" = some (fun i => i.1.val)
  offsam : current.regs .nat [p.blockRows] "offs_am" = some (fun i => row p initial i.1)
  fptrs : current.regs .ptr [p.blockRows,p.blockColumns] "fpa_ptrs" =
    some (fun i => (p.input, inputOffset p initial k i))
  aptrs : current.regs .ptr [p.blockRows,p.blockColumns] "a_ptrs" =
    some (fun i => (p.output, outputOffset p initial 0 i))
  maxval : current.regs .real [p.blockRows] "a_max" = some (fun i => runningMaximum p initial k i.1)

def preludeExecution (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (s : State Word) :
    {t : State Word // run (backend rest cfg u) (prelude p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [prelude, source, ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    bop, natLt, numericLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType, Region.cast, Option.bind,
    integralFloorDiv, integralMod, backend, QuantizeKvTransform.MixedBits.backend,
    QuantizeKvCopy.TypedQuantize.backend, fp32TotalBackend, fp32Source,
    FP.Software.scalarFP32, FP.Software.fp32Context]
  rfl

theorem preludeInvariant (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (s : State Word) :
    FirstInvariant p s 0 (preludeExecution rest cfg u p s).val := by
  constructor <;> simp [preludeExecution, row, inputOffset, outputOffset, runningMaximum,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]
  all_goals funext i
  all_goals simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]

def firstExecution (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (initial current : State Word) (k : Nat)
    (hw : 0 < p.blockColumns) (h : FirstInvariant p initial k current) :
    {t : State Word // run (backend rest cfg u) (firstBody p)
      (current.setReg "k" .nat [] (fun _ => k)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [firstBody, source, ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    bop, natLt, numericLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType, Region.cast, Option.bind,
    integralFloorDiv, integralMod, backend, QuantizeKvTransform.MixedBits.backend,
    QuantizeKvCopy.TypedQuantize.backend, fp32TotalBackend, fp32Source,
    FP.Software.scalarFP32, FP.Software.fp32Context,
    TileShape.axisDim, hw, h.memory, h.offsk, h.fptrs, h.maxval]
  rfl

theorem firstInvariant (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (initial current : State Word) (k : Nat)
    (hw : 0 < p.blockColumns) (h : FirstInvariant p initial k current) :
    FirstInvariant p initial (k+1) (firstExecution rest cfg u p initial current k hw h).val := by
  constructor <;>
    simp [firstExecution, h.memory, h.pidm, h.offsk, h.offsam, h.fptrs, h.aptrs, h.maxval,
      pairMaximum, runningMaximum, blockMaximum, inputOffset, outputOffset, loaded, active,
      QuantizeKvCopy.TypedQuantize.reduction, TileShape.insertAxisIndex,
      FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      Nat.add_mul, Nat.add_assoc, show (0.0 : ℝ) = 0 by norm_num]
  case fptrs =>
    funext i
    change (p.input, (row p initial i.1 * p.inputRowStride +
      (i.2.1.val * p.inputColumnStride + k * (p.blockColumns * p.inputColumnStride))) +
        p.blockColumns * p.inputColumnStride) = _
    simp [inputOffset, Nat.add_assoc, Nat.add_mul]
  case maxval =>
    funext i
    rfl

theorem firstLoop (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (initial current : State Word) (k : Nat) (hk : k ≤ p.steps)
    (hw : 0 < p.blockColumns) (h : FirstInvariant p initial k current) :
    ∃ final, FP.Structural.range (backend rest cfg u) "k" k p.steps 1 (firstBody p) current = some final ∧
      FirstInvariant p initial p.steps final := by
  have aux : ∀ n k (current : State Word), p.steps-k=n → k ≤ p.steps →
      FirstInvariant p initial k current →
      ∃ final, FP.Structural.range (backend rest cfg u) "k" k p.steps 1 (firstBody p) current = some final ∧
        FirstInvariant p initial p.steps final := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih =>
      intro k current hn hk h
      by_cases hlt : k < p.steps
      · let next := firstExecution rest cfg u p initial current k hw h
        have hi := firstInvariant rest cfg u p initial current k hw h
        obtain ⟨final, he, hf⟩ := ih (p.steps-(k+1)) (by omega) (k+1) next.val rfl (by omega) hi
        refine ⟨final, ?_, hf⟩
        rw [FP.Structural.range, if_neg (by decide), if_pos hlt, next.property]
        exact he
      · have heq : k=p.steps := by omega
        subst k
        refine ⟨current, ?_, h⟩
        rw [FP.Structural.range, if_neg (by decide), if_neg (by omega)]
  exact aux _ _ _ rfl hk h

#axiomsClean preludeExecution
#axiomsClean preludeInvariant
#axiomsClean firstExecution
#axiomsClean firstInvariant
#axiomsClean firstLoop
end
end VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPQuantize
