import bench.tests.WhileBody

namespace VeriTile.Tests.WhileBody
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 2000
set_option linter.unusedSimpArgs false

variable {α : Type} [Inhabited α] (A : Algebra α) (s : State α)

/-- The zero-trip path leaves the initial counter at zero. -/
theorem zero_iterations :
    (exec A (counter 0) s).bind (fun final => final.regs .nat [] "i") =
      some (fun _ => 0) := by
  simp [counter, exec, ComputeKernel.surfaceBody, run, step, evalExpr, evalOp_unfold,
    Op.natToUnsigned, Op.intToUnsigned, numeric, numericLt, natLt, integralMod,
    bop, Broadcast.leftIndex, Broadcast.rightIndex, Option.bind,
    State.setReg]
  repeat' (rw [Control.While.run]; simp [evalOp_unfold, numeric, numericLt, natLt,
    bop, integralMod, IntegerExpr.sameShape, Broadcast.leftIndex, Broadcast.rightIndex,
    State.setReg, Option.bind])

/-- A stale entry guard would never finish this program. -/
theorem reevaluates_guard :
    (exec A (counter 3) s).bind (fun final => final.regs .nat [] "i") =
      some (fun _ => 3) := by
  simp [counter, exec, ComputeKernel.surfaceBody, run, step, evalExpr, evalOp_unfold,
    Op.natToUnsigned, Op.intToUnsigned, numeric, numericLt, natLt, integralMod,
    bop, Broadcast.leftIndex, Broadcast.rightIndex, Option.bind,
    State.setReg]
  repeat' (rw [Control.While.run]; simp [evalOp_unfold, numeric, numericLt, natLt,
    bop, integralMod, IntegerExpr.sameShape, Broadcast.leftIndex, Broadcast.rightIndex,
    State.setReg, Option.bind])

/-- The loop stops after uint64 maximum wraps to zero, not after an arbitrary
fuel count and not at the unbounded mathematical successor. -/
theorem unsigned_wrap_stops :
    (exec A wrapping s).bind (fun final => final.regs .nat [] "i") =
      some (fun _ => 0) := by
  simp [wrapping, exec, ComputeKernel.surfaceBody, run, step, evalExpr, evalOp_unfold,
    Op.natToUnsigned, Op.intToUnsigned, numeric, numericLt, natLt, integralMod,
    bop, Broadcast.leftIndex, Broadcast.rightIndex, Option.bind,
    State.setReg]
  repeat' (rw [Control.While.run]; simp [evalOp_unfold, numeric, numericLt, natLt,
    bop, integralMod, IntegerExpr.sameShape, Broadcast.leftIndex, Broadcast.rightIndex,
    State.setReg, Option.bind])

#axiomsClean zero_iterations
#axiomsClean reevaluates_guard
#axiomsClean unsigned_wrap_stops
end
end VeriTile.Tests.WhileBody
