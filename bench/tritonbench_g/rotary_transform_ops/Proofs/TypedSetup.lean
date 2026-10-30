import bench.tritonbench_g.rotary_transform_ops.Proofs.SourceEntry
import bench.tritonbench_g.rotary_transform_ops.Kernels
import bench.tritonbench_g.rotary_transform_ops.Proofs.ProgramModel
import bench.tritonbench_g.isfinite_kernel.Proofs.Mathematical_MemoryProjection
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.RotaryTransformOps.TypedExecution
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits (fold_writes_regs)
noncomputable section
set_option maxHeartbeats 150000
set_option maxRecDepth 24000
set_option pp.maxSteps 8000
set_option linter.unusedSimpArgs false

def outputRegion (s : Shape) : RegionName := if s.inplace then "x" else "out"

/-- The canonical contiguous strides are the existing admitted launch strides.
Metadata values are read by the source, never supplied in this description. -/
def kernel (s : Shape) (cumulative64 offsets64 : Bool) : ComputeKernel :=
  KernelSource.source cumulative64 offsets64 (outputRegion s) "x" "cos" "sin" "cu" "offsets"
    s.scalarOffset s.maximum (2*s.half) s.coefficientRows
    (if s.varlen then 0 else s.maximum*s.heads*s.width) (s.heads*s.width) s.width 1
    (if s.varlen then 0 else s.maximum*s.heads*s.width) (s.heads*s.width) s.width 1
    s.blockK s.blockM s.offsetTensor s.varlen s.interleaved s.conjugate

/-- The six-statement source prefix evaluates program IDs, original signed
metadata and scalar input/output pointers. Its state is derived by execution
of the quotation, without supplied register or output values. -/
def setup {α : Type} [Inhabited α] (M : Algebra α) (s : Shape)
    (cumulative64 offsets64 : Bool) (initial : State α)
    :
    { final // FP.Structural.run M ((kernel s cumulative64 offsets64).surfaceBody.take 6) initial = some final } := by
  by_cases hv : s.varlen = Bool.true
  all_goals
    refine ⟨?_, ?_⟩
    swap
    unfold kernel
    rw [KernelSource.signed_carrier]
    simp (config := { maxSteps := 10000 }) [KernelSource.source64_64, hv,
      FP.Structural.exec, ComputeKernel.surfaceBody, FP.Structural.run, FP.Structural.step,
      evalExpr, evalComputeOp, evalOp_unfold,
      numeric, natLt, numericLt, numericLe, numericEq, bop, Region.cast,
      State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,
      State.setReg_undef, FP.Structural.store, TileShape.axisDim, TileShape.eraseAxis,
      TileShape.insertAxisIndex, integralFloorDiv, integralMod, ofFloat, toFloat,
      FloatDType.numericDType, FloatDType.toTileDType, fold_writes_regs, List.take, Option.bind,
      Option.isSome_some, Bool.false_eq_true, if_false, if_true, Bool.not_false, Bool.not_true,
      Option.bind_some,Option.some.injEq,Option.isSome_ite,
      Int.natCast_mul,ComputeDType.eraseDType,decide_eq_true_eq]
    try rfl

#axiomsClean setup

def ifBody : ComputeStmt → List ComputeStmt
  | .ifThen _ body => body
  | _ => []

def continuation (s : Shape) : List ComputeStmt :=
  ((kernel s Bool.true Bool.true).surfaceBody.drop 6).flatMap ifBody

/-- The next four assignments are extracted from the actual early-return body. -/
def indices {α : Type} [Inhabited α] (M : Algebra α) (s : Shape)
    (initial : State α) :
    { final // run M ((continuation s).take 4)
        (setup M s Bool.true Bool.true initial).val = some final } := by
  by_cases hv : s.varlen = Bool.true <;> by_cases ht : s.offsetTensor = Bool.true
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (config := { maxSteps := 20000 }) [continuation,ifBody,setup,kernel,
      KernelSource.source,KernelSource.source64_64, hv,ht,
      FP.Structural.exec, ComputeKernel.surfaceBody, FP.Structural.run, FP.Structural.step,
      evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numeric, natLt, numericLt, numericLe, numericEq, bop, Region.cast,
      State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,
      State.setReg_undef, FP.Structural.store, TileShape.axisDim, TileShape.eraseAxis,
      TileShape.insertAxisIndex, integralFloorDiv, integralMod, ofFloat, toFloat,
      FloatDType.numericDType, FloatDType.toTileDType, fold_writes_regs, List.take,
      ComputeDType.eraseDType]
    try rfl

#axiomsClean indices
end
end VeriTile.Bench.Repairs.RotaryTransformOps.TypedExecution
