import bench.tritonbench_g.int8_matmul_kernel.Proofs.TileReference
import VeriTile.Triton.Float.SignedCast
namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001 (Byte)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
attribute [local simp 2000] FP.Structural.eval_intToSigned
private theorem min_lt_form (a b : Nat) : min a b = if a < b then a else b := by
  by_cases h : a < b
  · rw [if_pos h, min_eq_left (Nat.le_of_lt h)]
  · rw [if_neg h, min_eq_right (Nat.le_of_not_gt h)]

structure Common (p : Parameters) (initial current : State Word32) : Prop where
  memory : current.mem = initial.mem
  pidm : current.regs .nat [] "pid_m" = some (fun _ => rowBlock p initial)
  pidn : current.regs .nat [] "pid_n" = some (fun _ => columnBlock p initial)
  offsk : current.regs .nat [p.blockInner] "offs_k" = some (fun i => i.1.val)
  offsn : current.regs .nat [p.blockColumns] "offs_bn" = some (fun i => inputColumn p initial i.1)
structure OuterInvariant (cast : Word32 → Byte) (p : Parameters)
    (initial : State Word32) (field : Nat) (current : State Word32) : Prop extends Common p initial current where
  aptrs : current.regs .ptr [p.blockRows,p.blockInner] "a_ptrs" =
    some (fun i => (p.input, aOffset p initial field 0 i))
  acc : current.regs .int [p.blockRows,p.blockColumns] "accumulator" =
    some (outerValues cast p initial field)
structure InnerInvariant (cast : Word32 → Byte) (p : Parameters)
    (initial : State Word32) (field step : Nat) (current : State Word32) : Prop extends Common p initial current where
  fieldReg : current.regs .nat [] "i" = some (fun _ => field)
  aptrs : current.regs .ptr [p.blockRows,p.blockInner] "a_ptrs" =
    some (fun i => (p.input, aOffset p initial field step i))
  bptrs : current.regs .ptr [p.blockInner,p.blockColumns] "b_ptrs" =
    some (fun i => (p.weight, bOffset p initial step i))
  acc : current.regs .int [p.blockRows,p.blockColumns] "accumulator" =
    some (innerValues cast p initial field (outerValues cast p initial field) step)

def preludeExecution (cast : Word32 → Byte) (M : Algebra Word32) (p : Parameters) (s : State Word32) :
    {t : State Word32 // run (backend cast M) (prelude p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [prelude, source, compileAssertionProjection, quotedSource, kernel, ComputeKernel.surfaceBody,
    run, step, stepAlg, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod]
  rfl

theorem preludeInvariant (cast : Word32 → Byte) (M : Algebra Word32) (p : Parameters) (s : State Word32) :
    OuterInvariant cast p s 0 (preludeExecution cast M p s).val := by
  refine ⟨?_, ?_, ?_⟩
  · constructor <;>
      simp [preludeExecution, backend, blocksM, blocksN, firstRow, rowsInGroup,
        rowBlock, columnBlock, inputColumn, inputRow, min_lt_form,
        FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]
    all_goals rfl
  · simp [preludeExecution, aOffset, inputRow, blocksM, blocksN, firstRow, rowsInGroup,
      rowBlock, columnBlock, min_lt_form, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]
    funext i
    simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]
  · simp [preludeExecution, outerValues]

def outerPreludeExecution (cast : Word32 → Byte) (M : Algebra Word32) (p : Parameters)
    (initial current : State Word32) (f : Nat) (h : OuterInvariant cast p initial f current) :
    {t : State Word32 // run (backend cast M) (outerPrelude p)
      (current.setReg "i" .nat [] (fun _ => f)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [outerPrelude, outerBody, source, compileAssertionProjection, quotedSource, kernel, ComputeKernel.surfaceBody,
    run, step, stepAlg, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
    bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod, h.offsk, h.offsn]
  rfl

theorem outerPreludeInvariant (cast : Word32 → Byte) (M : Algebra Word32) (p : Parameters)
    (initial current : State Word32) (f : Nat) (h : OuterInvariant cast p initial f current) :
    InnerInvariant cast p initial f 0 (outerPreludeExecution cast M p initial current f h).val := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · constructor <;> simp [outerPreludeExecution, h.memory, h.pidm, h.pidn, h.offsk, h.offsn]
  · simp [outerPreludeExecution]
  · simp [outerPreludeExecution, h.aptrs]
  · simp [outerPreludeExecution, bOffset, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]
    funext i
    simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]
  · simp [outerPreludeExecution, innerValues, h.acc]

#axiomsClean preludeExecution
#axiomsClean preludeInvariant
#axiomsClean outerPreludeExecution
#axiomsClean outerPreludeInvariant
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
