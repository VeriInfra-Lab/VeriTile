import bench.tritonbench_g.int8_dequant_matmul.Proofs.FPReference
namespace VeriTile.Bench.TritonBenchG.Int8DequantMatmul.FPTyped
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

private theorem min_lt_form (a b : Nat) : min a b = if a < b then a else b := by
  by_cases h : a < b
  · rw [if_pos h, min_eq_left (Nat.le_of_lt h)]
  · rw [if_neg h, min_eq_right (Nat.le_of_not_gt h)]

structure Invariant {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (initial : State α) (divisor : α) (k : Nat) (current : State α) : Prop where
  memory : current.mem = initial.mem
  rm : current.regs .nat [p.blockRows] "rm" =
    some (fun i => rowBlock p initial * p.blockRows + i.1.val)
  rn : current.regs .nat [p.blockColumns] "rn" =
    some (fun i => columnBlock p initial * p.blockColumns + i.1.val)
  factor : current.regs .fp32 [] "divfactor" = some (fun _ => divisor)
  wfactor : current.regs .real [1,p.blockColumns] "w_factor" =
    some (fun i => columnFactor M p initial i.2.1)
  xfactor : current.regs .real [p.blockRows,1] "x_factor" =
    some (fun i => rowFactor M p initial i.1)
  aptrs : current.regs .ptr [p.blockRows,p.blockInner] "A" =
    some (fun i => (p.input, aOffset p initial k i))
  bptrs : current.regs .ptr [p.blockInner,p.blockColumns] "B" =
    some (fun i => (p.weight, bOffset p initial k i))
  acc : current.regs .int [p.blockRows,p.blockColumns] "acc_int" =
    some (accumulation p initial k)

def preludeExecution {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    {t : State α // run (backend M) (prelude p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [prelude, original, typed, scales, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, eval_intToSigned, evalOffsets, store, numeric,
    bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod]
  rfl

theorem preludeInvariant {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) (divisor : α)
    (hdiv : s.regs .fp32 [] "divfactor" = some (fun _ => divisor)) :
    Invariant M p s divisor 0 (preludeExecution M p s).val := by
  constructor <;>
    simp [preludeExecution, backend, accumulation, blocksM, blocksN, firstRow,
      rowsInGroup, rowBlock, columnBlock, inputInner, inputColumn, inputRow,
      aOffset, bOffset, rowFactor, columnFactor, min_lt_form, hdiv,
      FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]
  all_goals funext i
  all_goals simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]

def bodyExecution {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (initial current : State α) (divisor : α) (k : Nat)
    (h : Invariant M p initial divisor k current) :
    {t : State α // run (backend M) (body p)
      (current.setReg "k" .nat [] (fun _ => k)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [body, originalBody, original, typed, widths, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, eval_intToSigned, evalOffsets, store, numeric,
    bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod,
    h.aptrs, h.bptrs, h.acc, h.memory, backend]
  rfl

private theorem expandedWrap (w : Nat) (x : Int) :
 (if 2 * ((x % (2^w : Int)).toNat % 2^w) < 2^w then
    max (x % (2^w : Int)) 0 % (2^w : Int)
  else max (x % (2^w : Int)) 0 % (2^w : Int) - (2^w : Int)) = wrap w x := by
 simpa [IntegerExpr.signedBits, wrap] using signedBits_int_mod w x

theorem bodyInvariant {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (initial current : State α) (divisor : α) (k : Nat)
    (h : Invariant M p initial divisor k current) :
    Invariant M p initial divisor (k + 1) (bodyExecution M p initial current divisor k h).val := by
  constructor <;>
    simp [bodyExecution, h.memory, h.rm, h.rn, h.factor, h.wfactor, h.xfactor,
      h.aptrs, h.bptrs, h.acc, backend, accumulation, aValue, bValue, expandedWrap,
      aOffset, bOffset,
      FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Nat.add_mul, Nat.add_assoc]
  all_goals funext i
  all_goals try simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, aValue, aOffset, Nat.add_assoc]
  all_goals rcases i with ⟨r, c, u⟩
  all_goals cases u
  all_goals delta aValue bValue
  all_goals simp only [aOffset, bOffset, Nat.add_assoc]
  all_goals
    have h8 (x : Int) : (if 2 * ((x % 256).toNat % 256) < 256 then max (x % 256) 0 % 256
      else max (x % 256) 0 % 256 - 256) = wrap 8 x := by simpa using expandedWrap 8 x
    have h32 (x : Int) : (if 2 * ((x % 4294967296).toNat % 4294967296) < 4294967296 then
      max (x % 4294967296) 0 % 4294967296 else max (x % 4294967296) 0 % 4294967296 - 4294967296) = wrap 32 x := by
      simpa using expandedWrap 32 x
    simp only [h8, h32]

#axiomsClean preludeExecution
#axiomsClean preludeInvariant
#axiomsClean bodyExecution
#axiomsClean bodyInvariant
end
end VeriTile.Bench.TritonBenchG.Int8DequantMatmul.FPTyped
