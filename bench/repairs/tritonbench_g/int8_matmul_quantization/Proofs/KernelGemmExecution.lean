import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.KernelGemmSource

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelGemm
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
variable {I F H : Type} [Inhabited F]

private theorem min_lt_form (a b : Nat) : min a b = if a < b then a else b := by
  by_cases h : a < b
  · rw [if_pos h, min_eq_left (Nat.le_of_lt h)]
  · rw [if_neg h, min_eq_right (Nat.le_of_not_gt h)]

def preludeExecution (M : Algebra F) (carrier : IntegerCarrier I)
    (ops : Int8DequantMatmul.Ops I F H) (plans : Plans)
    (rowScale output : Precision) (p : Parameters) (s : State F) :
    {t : State F // KernelExecution.run M carrier ops plans (prelude rowScale output p) s = some t} := by
  cases rowScale <;> cases output
  all_goals (refine ⟨?_, ?_⟩; swap
             simp [prelude, source, KernelSource.gemmHalfHalf, KernelSource.gemmHalfSingle,
    KernelSource.gemmSingleHalf, KernelSource.gemmSingleSingle, ComputeKernel.surfaceBody,
    KernelExecution.run, KernelExecution.step, localStep, FP.Structural.step,
    evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType, Region.cast, Option.bind,
    integralFloorDiv, integralMod]
             rfl)

theorem preludeInvariant (M : Algebra F) (carrier : IntegerCarrier I)
    (ops : Int8DequantMatmul.Ops I F H) (plans : Plans)
    (rowScale output : Precision) (p : Parameters) (s : State F) :
    Invariant M carrier ops plans rowScale p s 0
      (preludeExecution M carrier ops plans rowScale output p s).val := by
  cases rowScale <;> cases output
  all_goals constructor
  all_goals simp [preludeExecution, accumulation, blocksM, blocksN, firstRow,
    rowsInGroup, rowBlock, columnBlock, inputInner, inputColumn, inputRow,
    aOffset, bOffset, aScale, bScale, min_lt_form, FP.Structural.bop,
    Broadcast.leftIndex, Broadcast.rightIndex, show (0.0 : ℝ)=0 by norm_num]
  all_goals funext i
  all_goals try simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]

def bodyExecution (M : Algebra F) (carrier : IntegerCarrier I)
    (ops : Int8DequantMatmul.Ops I F H) (plans : Plans)
    (rowScale : Precision) (p : Parameters) (initial current : State F) (k : Nat)
    (h : Invariant M carrier ops plans rowScale p initial k current) :
    {t : State F // KernelExecution.run M carrier ops plans (body p)
      (current.setReg "k" .nat [] (fun _ => k)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [body, source, KernelSource.gemmHalfHalf, ComputeKernel.surfaceBody,
    KernelExecution.run, KernelExecution.step, localStep_dot, localStep_load,
    localStep_pointer, accumulate, FP.Structural.step,
    evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType, Region.cast, Option.bind,
    integralFloorDiv, integralMod, h.offsk, h.aptrs, h.bptrs, h.acc, h.memory,
    h.pidm, h.pidn, h.pids]
  rfl

theorem bodyInvariant (M : Algebra F) (carrier : IntegerCarrier I)
    (ops : Int8DequantMatmul.Ops I F H) (plans : Plans)
    (rowScale : Precision) (p : Parameters) (initial current : State F) (k : Nat)
    (h : Invariant M carrier ops plans rowScale p initial k current) :
    Invariant M carrier ops plans rowScale p initial (k+1)
      (bodyExecution M carrier ops plans rowScale p initial current k h).val := by
  constructor
  all_goals simp [bodyExecution, h.memory, h.pids, h.pidm, h.pidn, h.offsk, h.aptrs,
    h.bptrs, h.ascale, h.bscale, h.acc, accumulation, aValue, bValue,
    aOffset, bOffset, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    Nat.add_mul, Nat.add_assoc]
  all_goals try (cases rowScale <;> simp_all)
  all_goals funext i
  all_goals try simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    aOffset, bOffset, Nat.add_assoc]

#axiomsClean preludeExecution
#axiomsClean preludeInvariant
#axiomsClean bodyExecution
#axiomsClean bodyInvariant
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelGemm
