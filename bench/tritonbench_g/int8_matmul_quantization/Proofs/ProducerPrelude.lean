import bench.tritonbench_g.int8_matmul_quantization.Proofs.ControlArithmetic
import bench.tritonbench_g.int8_matmul_quantization.Proofs.Sources.KernelSourceBridge
import bench.tritonbench_g.kcache_copy_triton.Proofs.Sources.Mathematical_TypedCopy

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer
open VeriTile.Triton FP.Structural
open ControlArithmetic
attribute [local simp 2000] FP.Structural.eval_truncDiv FP.Structural.eval_truncMod
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

/-- The launched producer has one row and a positive lane width. Its actual
loop count is derived from K and the lane width; no output is supplied. -/
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

theorem fold_regs {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active] (r : RegionName)
    (off : ι → Nat) (v : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => if active i then t.write r (off i) (v i) else t) s).regs = s.regs := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih => simp only [List.foldl_cons, ih]; split <;> rfl

theorem fold_pids {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active] (r : RegionName)
    (off : ι → Nat) (v : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => if active i then t.write r (off i) (v i) else t) s).pids = s.pids := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih => simp only [List.foldl_cons, ih]; split <;> rfl


def code (p : Parameters) : List ComputeStmt :=
  (KernelSource.quantize (parameters p)).surfaceBody

def prelude (p : Parameters) : List ComputeStmt := (code p).take 13

def maxBody (p : Parameters) : List ComputeStmt :=
  match (code p)[13]? with
  | some (ComputeStmt.forRangeInt _ _ _ _ body) => body
  | _ => []

def quantizeBody (p : Parameters) : List ComputeStmt :=
  match (code p)[16]? with
  | some (ComputeStmt.forRangeInt _ _ _ _ body) => body
  | _ => []

def tripCount (p : Parameters) : Nat := (p.columns+p.width-1)/p.width

/-- All seven explicit metadata casts, the signed program id, wrapped row,
pointers and zero initialization execute before either source loop. -/
def prefixExecution {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α → BitVec 8)
    (p : Parameters) (s : State α) :
    {t : State α // run (backend M integer8) (prelude p) s=some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [prelude, code, KernelSource.quantize, quantizeKernel, parameters,
    ComputeKernel.surfaceBody, List.take_succ_cons, List.take_zero]
  simp [run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, numeric,
    bop, backend, ComputeDType.eraseDType, Option.bind, NumericDType.sub,
    Broadcast.leftIndex, Broadcast.rightIndex, integralFloorDiv, integralMod]
  rfl

#axiomsClean prefixExecution
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer
