import bench.tritonbench_g.rotary_transform.Proofs.TypedSetup

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.TypedExecution
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits (fold_writes_regs)
noncomputable section
set_option maxHeartbeats 300000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def branchBody (interleaved : Bool) : ComputeStmt → List ComputeStmt
  | .ifThenElse _ split inter => if interleaved then inter else split
  | _ => []

def payload (s : Shape) : List ComputeStmt :=
  ((continuation s).drop (TypedExecution.indicesCount s)).flatMap (branchBody s.interleaved)

def loadsCount (s : Shape) : Nat := if s.interleaved then 14 else 11

/-- Complete loads, before either source store. -/
def loads {α : Type} [Inhabited α] (M : Algebra α) (s : Shape)
    (initial : State α) :
    { final // run M ((payload s).take (loadsCount s))
        (indices M s initial).val = some final } := by
  by_cases hv : s.varlen = Bool.true <;> by_cases ht : s.offsetTensor = Bool.true <;>
    by_cases hi : s.interleaved = Bool.true
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (config := { maxSteps := 40000 }) [payload,loadsCount,branchBody,continuation,ifBody,
      indices,setup,kernel,KernelSource.source,KernelSource.source64_64, hv,ht,hi,
      FP.Structural.exec, VeriTile.Bench.Repairs.RotaryTransformTileChecked.originalTensorKernel,
      VeriTile.Bench.Repairs.RotaryTransformTileChecked.originalScalarKernel,
      VeriTile.Bench.Repairs.RotaryTransformTileChecked.opsOriginalTensorKernel,
      VeriTile.Bench.Repairs.RotaryTransformTileChecked.opsOriginalScalarKernel,
      VeriTile.Bench.Repairs.RotaryTransformTileChecked.candidateTensorKernel,
      VeriTile.Bench.Repairs.RotaryTransformTileChecked.candidateScalarKernel,
      VeriTile.Bench.Repairs.RotaryTransformTileChecked.xorTensorKernel,
      VeriTile.Bench.Repairs.RotaryTransformTileChecked.xorScalarKernel,ComputeKernel.withScalarArguments,ScalarArgument.statement,ScalarArgument.integer,TypedExecution.indicesCount,ComputeKernel.surfaceBody, FP.Structural.run, FP.Structural.step,
      evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numeric, natLt, numericLt, numericLe, numericEq, bop, Region.cast,
      State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,
      State.setReg_undef, FP.Structural.store, TileShape.axisDim, TileShape.eraseAxis,
      TileShape.insertAxisIndex, integralFloorDiv, integralMod, ofFloat, toFloat,
      FloatDType.numericDType, FloatDType.toTileDType, fold_writes_regs, List.take,
      ComputeDType.eraseDType]
    try rfl

#axiomsClean loads
end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.TypedExecution
