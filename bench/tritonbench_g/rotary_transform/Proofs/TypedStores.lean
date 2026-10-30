import bench.tritonbench_g.rotary_transform.Proofs.TypedLoads

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.TypedExecution
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits (fold_writes_regs)
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false


/-- The remaining six source statements conjugate, compute and store from
already loaded registers. Both split stores use the same pre-store operands. -/
def finish {α : Type} [Inhabited α] (M : Algebra α) (s : Shape)
    (initial : State α) :
    { final // run M ((payload s).drop (loadsCount s))
        (loads M s initial).val = some final } := by
  by_cases hv : s.varlen = Bool.true <;> by_cases ht : s.offsetTensor = Bool.true <;>
    by_cases hi : s.interleaved = Bool.true <;> by_cases hc : s.conjugate = Bool.true
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (config := { maxSteps := 50000 }) [payload,loadsCount,branchBody,continuation,ifBody,
      loads,indices,setup,kernel,KernelSource.source,KernelSource.source64_64, hv,ht,hi,hc,
      FP.Structural.exec, ComputeKernel.surfaceBody, FP.Structural.run, FP.Structural.step,
      evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
      numeric, natLt, numericLt, numericLe, numericEq, bop, Region.cast,
      ↓State.setReg_same,↓State.setReg_ne_name,↓State.setReg_mem,↓State.setReg_pids,
      ↓State.setReg_undef, FP.Structural.store, TileShape.axisDim, TileShape.eraseAxis,
      TileShape.insertAxisIndex, integralFloorDiv, integralMod, ofFloat, toFloat,
      FloatDType.numericDType, FloatDType.toTileDType, ↓fold_writes_regs, List.take,
      ComputeDType.eraseDType]
    try rfl

#axiomsClean finish
end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.TypedExecution
