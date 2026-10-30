import bench.tritonbench_g.int8_matmul_quantization.Sources.KernelSourceBridge
import bench.tritonbench_g.kcache_copy_triton.Sources.Mathematical_TypedCopy

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

/-- The actual checked producer specialization has BM=1 and BK>=K>0,
hence exactly one iteration of each source loop. These parameters record
strides and sizes only; no desired value or output memory is an input. -/
structure Parameters where
  (input output scales : RegionName)
  (rows columns width inputRowStride inputColumnStride outputRowStride
    outputColumnStride scaleStride : Nat)

def parameters (p : Parameters) : KernelSource.QuantizeParameters where
  input := p.input
  output := p.output
  scales := p.scales
  rows := p.rows
  columns := p.columns
  inputRowStride := p.inputRowStride
  inputColumnStride := p.inputColumnStride
  outputRowStride := p.outputRowStride
  outputColumnStride := p.outputColumnStride
  scaleStride := p.scaleStride
  blockRows := 1
  blockColumns := p.width
  steps := 1

/-- The observed signed16 conversion followed by low8 storage is supplied
as one total scalar instruction. It is not a finite-input guard or an
assumption about a row result. All other arithmetic is the shared algebra. -/
def backend (M : Algebra α) (integer8 : α → BitVec 8) : Algebra α :=
  { M with floatToInt := fun pr d width =>
      if width = 8 then some (fun x => some (integer8 x).toInt)
      else M.floatToInt pr d width }

private theorem fold_regs {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active] (r : RegionName)
    (off : ι → Nat) (v : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => if active i then t.write r (off i) (v i) else t) s).regs = s.regs := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih => simp only [List.foldl_cons, ih]; split <;> rfl

private theorem fold_pids {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active] (r : RegionName)
    (off : ι → Nat) (v : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => if active i then t.write r (off i) (v i) else t) s).pids = s.pids := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih => simp only [List.foldl_cons, ih]; split <;> rfl

/-- Successful execution of the entire quoted producer, including both
loops and both stores, without a numerical-domain premise. -/
def execution {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α → BitVec 8)
    (p : Parameters) (s : State α) (hw : 0 < p.width) :
    {t : State α // FP.Structural.exec (backend M integer8)
      (KernelSource.quantize (parameters p)) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp (config := { maxSteps := 1000000 }) [KernelSource.quantize, parameters,
    FP.Structural.exec, ComputeKernel.surfaceBody, run, step, range, evalExpr,
    evalComputeOp, evalOp_unfold, evalOffsets, store, numeric, natLt, bop,
    numericEq, backend, State.setReg, ComputeDType.eraseDType, ofFloat, toFloat,
    Region.cast, Option.bind, TileShape.axisDim, hw, fold_regs, fold_pids,
    integralFloorDiv, integralMod]
  try rfl

#axiomsClean execution
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer
