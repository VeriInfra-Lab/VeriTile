import bench.tritonbench_g.matmul_dequantize_int4.Sources.Mathematical_FPReference
namespace VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPTyped
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 32000
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
  numk : current.regs .nat [] "num_pid_k" = some (fun _ => steps p)
  bits : current.regs .nat [] "bits" = some (fun _ => 4)
  inf : current.regs .nat [] "infearure_per_bits" = some (fun _ => 8)
  shift : current.regs .nat [p.blockInner] "shifter" = some (fun i => i.1.val % 8 * 4)
  zshift : current.regs .nat [p.blockColumns] "zeros_shifter" =
    some (fun i => inputColumn p initial i.1 % 8 * 4)
  amask : current.regs .bool [p.blockRows,1] "a_mask" =
    some (fun i => decide (inputRow p initial i.1 < p.rows))
  aptrs : current.regs .ptr [p.blockRows,p.blockInner] "a_ptrs" =
    some (fun i => (p.input, aOffset p initial k i))
  bptrs : current.regs .ptr [p.blockInner,p.blockColumns] "b_ptrs" =
    some (fun i => (p.weight, bOffset p initial k i))
  sptrs : current.regs .ptr [p.blockColumns] "scales_ptrs" =
    some (fun i => (p.scale, inputColumn p initial i.1 * p.scaleColumnStride))
  zptrs : current.regs .ptr [p.blockColumns] "zeros_ptrs" =
    some (fun i => (p.zero, inputColumn p initial i.1 / 8 * p.zeroColumnStride))
  scales : p.noGroups = Bool.true → current.regs .fp16 [p.blockColumns] "scales" =
    some (scaleValue p initial 0)
  zeros : p.noGroups = Bool.true → current.regs .fp16 [p.blockColumns] "zeros" =
    some (zeroValue M p initial 0)
  acc : current.regs .real [p.blockRows,p.blockColumns] "accumulator" =
    some (accumulation M D p initial k)

def preludeExecution {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (s : State α) :
    {t : State α // run (backend M D) (prelude p) s = some t} := by
  by_cases hno : p.noGroups = Bool.true <;>
    (refine ⟨?_, ?_⟩
     swap
     simp [prelude, source, typed, ComputeKernel.surfaceBody,
       run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
       bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
       FloatDType.numericDType, FloatDType.toTileDType,
       Region.cast, Option.bind, integralFloorDiv, integralMod, hno]
     rfl)

theorem preludeInvariant {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (s : State α) :
    Invariant M D p s 0 (preludeExecution M D p s).val := by
  have hmask (n : Nat) : n &&& 15 < 4294967296 :=
    lt_of_le_of_lt Nat.and_le_right (by decide)
  have hmaskInt (n : Nat) : ((n &&& 15 : Nat) : Int) % 4294967296 = ((n &&& 15 : Nat) : Int) := by
    apply Int.emod_eq_of_lt (by omega)
    exact_mod_cast hmask n
  cases hno : p.noGroups
  all_goals constructor <;>
    simp [preludeExecution, hno, backend, accumulation, blocksM, blocksN, firstRow,
      rowsInGroup, rowBlock, columnBlock, inputInner, inputColumn, inputRow, steps,
      aOffset, bOffset, scaleOffset, zeroOffset, scaleValue, zeroValue, zeroNibble, groupIndex,
      min_lt_form, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      MatmulDequantInt4.FPTyped.nibble,
      BitVec.toInt_eq_toNat_cond, BitVec.toNat_and, BitVec.toNat_ofInt,
      Int.shiftRight_eq_div_pow, Nat.shiftLeft_eq, Int.shiftLeft_eq,
      Nat.mod_eq_of_lt (hmask _), hmaskInt]
  all_goals funext i
  all_goals try simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    scaleValue, zeroValue, zeroNibble, scaleOffset, zeroOffset, groupIndex, hno,
    inputColumn, columnBlock, rowsInGroup, firstRow, blocksM, blocksN, min_lt_form,
    MatmulDequantInt4.FPTyped.nibble, BitVec.toInt_eq_toNat_cond, BitVec.toNat_and,
    BitVec.toNat_ofInt, Int.shiftRight_eq_div_pow, Nat.shiftLeft_eq, Int.shiftLeft_eq,
    Nat.mod_eq_of_lt (hmask _), hmaskInt]
  all_goals rfl

def bodyExecution {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (initial current : State α) (k : Nat)
    (h : Invariant M D p initial k current) :
    {t : State α // run (backend M D) (body p)
      (current.setReg "k" .nat [] (fun _ => k)) = some t} := by
  by_cases hno : p.noGroups = Bool.true <;>
    (refine ⟨?_, ?_⟩
     swap
     simp [body, source, typed, ComputeKernel.surfaceBody,
       run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
       bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
       FloatDType.numericDType, FloatDType.toTileDType,
       Region.cast, Option.bind, integralFloorDiv, integralMod,
       h.aptrs, h.bptrs, h.acc, h.memory, h.amask, h.sptrs, h.zptrs, h.shift, h.zshift,
       h.bits, h.inf, h.scales, h.zeros, hno, backend]
     rfl)

theorem bodyInvariant {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (initial current : State α) (k : Nat)
    (h : Invariant M D p initial k current) :
    Invariant M D p initial (k + 1) (bodyExecution M D p initial current k h).val := by
  have hmask (n : Nat) : n &&& 15 < 4294967296 :=
    lt_of_le_of_lt Nat.and_le_right (by decide)
  have hmaskInt (n : Nat) : ((n &&& 15 : Nat) : Int) % 4294967296 = ((n &&& 15 : Nat) : Int) := by
    apply Int.emod_eq_of_lt (by omega)
    exact_mod_cast hmask n
  cases hno : p.noGroups
  all_goals constructor <;>
    simp [bodyExecution, h.memory, h.pidm, h.pidn, h.numk, h.bits, h.inf, h.shift, h.zshift,
      h.amask, h.aptrs, h.bptrs, h.sptrs, h.zptrs, h.scales, h.zeros, h.acc, hno,
      backend, accumulation, aValue, bValue, weightNibble, zeroNibble, scaleValue, zeroValue,
      aOffset, bOffset, scaleOffset, zeroOffset, groupIndex, inputInner,
      MatmulDequantInt4.FPTyped.nibble,
      BitVec.toInt_eq_toNat_cond, BitVec.toNat_and, BitVec.toNat_ofInt,
      Int.shiftRight_eq_div_pow, Nat.shiftLeft_eq, Int.shiftLeft_eq,
      Nat.mod_eq_of_lt (hmask _), hmaskInt,
      FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Nat.add_mul, Nat.add_assoc]
  all_goals funext i
  all_goals simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Nat.add_assoc]
  all_goals rcases i with ⟨r,c,u⟩
  all_goals cases u
  all_goals rw [show aValue M p initial k = (fun i =>
    if inputRow p initial i.1 < p.rows then
      (initial.mem p.input (inputRow p initial i.1 * p.inputRowStride +
        (i.2.1.val * p.inputInnerStride + k * (p.blockInner * p.inputInnerStride)))).read .fp16
    else M.cast none .real .fp16 (M.literal none .real 0.0)) from by
      funext j
      simp [aValue, aOffset, inputInner, Nat.add_assoc]
      norm_num]
  all_goals rfl

#axiomsClean preludeExecution
#axiomsClean preludeInvariant
#axiomsClean bodyExecution
#axiomsClean bodyInvariant
end
end VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPTyped
