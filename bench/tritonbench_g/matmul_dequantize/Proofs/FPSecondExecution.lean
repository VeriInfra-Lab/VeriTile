import bench.tritonbench_g.matmul_dequantize.Proofs.FPSecondReference
namespace VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedSecond
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

private theorem min_lt_form (a b : Nat) : min a b = if a < b then a else b := by
  by_cases h : a < b
  · rw [if_pos h, min_eq_left (Nat.le_of_lt h)]
  · rw [if_neg h, min_eq_right (Nat.le_of_not_gt h)]

structure Invariant {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (initial : State α) (k : Nat) (current : State α) : Prop where
  memory : current.mem = initial.mem
  pidm : current.regs .nat [] "pid_m" = some (fun _ => rowBlock p initial)
  pidn : current.regs .nat [] "pid_n" = some (fun _ => columnBlock p initial)
  offsk : current.regs .nat [p.blockInner] "offs_k" = some (fun i => inputInner p initial i.1)
  offsn : current.regs .nat [p.blockColumns] "offs_bn" = some (fun i => inputColumn p initial i.1)
  aptrs : current.regs .ptr [p.blockRows,p.blockInner] "a_ptrs" =
    some (fun i => (p.input, aOffset p initial k i))
  bptrs : current.regs .ptr [p.blockInner,p.blockColumns] "b_ptrs" =
    some (fun i => (p.weight, bOffset p initial k i))
  acc : current.regs .real [p.blockRows,p.blockColumns] "accumulator" =
    some (accumulation M D p initial k)

def preludeExecution {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (s : State α) :
    {t : State α // run (backend M D) (prelude p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [prelude, source, FPTyped.matmulTyped, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod]
  rfl

theorem preludeInvariant {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (s : State α) :
    Invariant M D p s 0 (preludeExecution M D p s).val := by
  constructor <;>
    simp [preludeExecution, backend, accumulation, blocksM, blocksN, firstRow,
      rowsInGroup, rowBlock, columnBlock, inputInner, inputColumn, inputRow,
      aOffset, bOffset, min_lt_form, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]
  all_goals funext i
  all_goals simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]

def bodyExecution {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (initial current : State α) (k : Nat)
    (h : Invariant M D p initial k current) :
    {t : State α // run (backend M D) (body p)
      (current.setReg "k" .nat [] (fun _ => k)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [body, source, FPTyped.matmulTyped, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod,
    h.offsk, h.offsn, h.aptrs, h.bptrs, h.acc, h.memory, backend]
  rfl

theorem bodyInvariant {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (initial current : State α) (k : Nat)
    (h : Invariant M D p initial k current) :
    Invariant M D p initial (k + 1) (bodyExecution M D p initial current k h).val := by
  have hmask (n : Nat) : n &&& 15 < 4294967296 :=
    lt_of_le_of_lt Nat.and_le_right (by decide)
  have hmaskInt (n : Nat) : ((n &&& 15 : Nat) : Int) % 4294967296 = ((n &&& 15 : Nat) : Int) := by
    apply Int.emod_eq_of_lt (by omega)
    exact_mod_cast hmask n
  constructor <;>
    simp [bodyExecution, h.memory, h.pidm, h.pidn, h.offsk, h.offsn,
      h.aptrs, h.bptrs, h.acc, backend, accumulation, aValue, bValue, difference,
      aOffset, bOffset, scaleOffset, zeroOffset,
      MatmulDequantInt4.FPTyped.nibble,
      BitVec.toInt_eq_toNat_cond, BitVec.toNat_and, BitVec.toNat_ofInt,
      Int.shiftRight_eq_div_pow, Nat.shiftLeft_eq, Int.shiftLeft_eq,
      Nat.mod_eq_of_lt (hmask _), hmaskInt,
      FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Nat.add_mul, Nat.add_assoc]
  all_goals funext i
  all_goals simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, aValue, aOffset, Nat.add_assoc]
  all_goals rcases i with ⟨r, c, u⟩
  all_goals cases u
  all_goals dsimp only [aValue, aOffset]
  all_goals try simp only [Nat.add_assoc]
  all_goals rw [show aValue M p initial k = (fun i =>
    M.cast (some .fp32) .fp16 .fp16 ((initial.mem p.input (inputRow p initial i.1 * p.inputRowStride +
      (inputInner p initial i.2.1 * p.inputInnerStride + k * (p.blockInner * p.inputInnerStride)))).read .fp16)) from by
        funext j
        simp [aValue, aOffset, Nat.add_assoc]]
  all_goals rfl

#axiomsClean preludeExecution
#axiomsClean preludeInvariant
#axiomsClean bodyExecution
#axiomsClean bodyInvariant
end
end VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedSecond
