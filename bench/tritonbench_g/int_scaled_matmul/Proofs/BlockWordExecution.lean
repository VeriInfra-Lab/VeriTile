import bench.tritonbench_g.int_scaled_matmul.Sources.BlockWordReference
namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.BlockWord
open VeriTile.Bench.TritonBenchG.IntScaledMatmul
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

private theorem min_lt_form (a b : Nat) : min a b = if a < b then a else b := by
  by_cases h : a < b
  · rw [if_pos h, min_eq_left (Nat.le_of_lt h)]
  · rw [if_neg h, min_eq_right (Nat.le_of_not_gt h)]

structure Invariant {α : Type} [Inhabited α]
    (p : Parameters) (initial : State α) (k : Nat) (current : State α) : Prop where
  memory : current.mem = initial.mem
  undefined : current.undef = initial.undef
  pidm : current.regs .nat [] "pid_m" = some (fun _ => rowBlock p initial)
  pidn : current.regs .nat [] "pid_n" = some (fun _ => columnBlock p initial)
  aptrs : current.regs .blockPtr [p.blockRows,p.blockInner] "a_block_ptr" =
    some (fun _ => aPointer p initial k)
  bptrs : current.regs .blockPtr [p.blockInner,p.blockColumns] "b_block_ptr" =
    some (fun _ => bPointer p initial k)
  acc : current.regs .int [p.blockRows,p.blockColumns] "accumulator" =
    some (accumulation p initial k)

def preludeExecution {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    {t : State α // run (backend M) (prelude p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [prelude, original, int_scaled_matmul_matmul_surface, VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.unscaled,
    VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.unscaledSource,
    VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.scaled,
    VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.scaledSource,ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, eval_intToSigned, evalOffsets, store, numeric,
    bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod]
  rfl

theorem preludeInvariant {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    Invariant p s 0 (preludeExecution M p s).val := by
  constructor <;>
    simp [preludeExecution, backend, accumulation, blocksM, blocksN, firstRow,
      rowsInGroup, rowBlock, columnBlock, aPointer, bPointer, min_lt_form,
      FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]
  all_goals funext i
  all_goals simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]

def bodyExecution {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (initial current : State α) (k : Nat)
    (h : Invariant p initial k current) :
    {t : State α // run (backend M) (body p)
      (current.setReg "k" .nat [] (fun _ => k * p.blockInner)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [body, originalBody, original, int_scaled_matmul_matmul_surface, VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.unscaled,
    VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.unscaledSource,
    VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.scaled,
    VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.scaledSource,ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, eval_intToSigned, evalOffsets, store, numeric,
    bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod,
    h.aptrs, h.bptrs, h.acc, h.memory, h.undefined, backend]
  rfl

private theorem expandedWrap (w : Nat) (x : Int) :
 (if 2 * ((x % (2^w : Int)).toNat % 2^w) < 2^w then
    max (x % (2^w : Int)) 0 % (2^w : Int)
  else max (x % (2^w : Int)) 0 % (2^w : Int) - (2^w : Int)) = wrap w x := by
 simpa [IntegerExpr.signedBits, wrap] using signedBits_int_mod w x

theorem bodyInvariant {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (initial current : State α) (k : Nat)
    (h : Invariant p initial k current) :
    Invariant p initial (k + 1) (bodyExecution M p initial current k h).val := by
  constructor <;>
    simp [bodyExecution, h.memory, h.undefined, h.pidm, h.pidn,
      h.aptrs, h.bptrs, h.acc, backend, accumulation, aValue, bValue, loaded,
      aPointer, bPointer, boundaryZero, defaultValue,
      FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Nat.add_mul, Nat.add_assoc]
  all_goals funext i
  all_goals rcases i with ⟨r, c, u⟩
  all_goals cases u
  all_goals try simp only [← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast]
  all_goals unfold loaded
  all_goals simp only [TileShape.indexToList, BlockPtr.inBounds_2d_offsets, BlockPtr.address_2d_offsets,
    Nat.zero_add, decide_eq_true_eq, Nat.add_mul, Nat.add_assoc]
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
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.BlockWord
