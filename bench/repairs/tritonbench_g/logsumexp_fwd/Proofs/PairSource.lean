import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.Engine
import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.AccessSafety

namespace VeriTile.Bench.Repairs.LogsumexpFwd.PairSource
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- The select is removed only where the exact typed masked load already
produces the same -infinity value. No arithmetic identity or finite-value
assumption is used, and the scaled nonfinite path is unchanged. -/
theorem exact [Inhabited A] (engine : Engine A) (x z : RegionName)
    (D B : Nat) (scaled : Bool) (s : State A) :
    exec engine.algebra (rowBlock x z D B scaled) s =
      exec engine.algebra (rowBlockCandidate x z D B scaled) s := by
  cases scaled
  all_goals cases hp : s.regs .real [] "scale"
  all_goals simp +contextual (maxSteps := 1800000) [rowBlock, rowBlockCandidate, FP.Structural.exec,
    ComputeKernel.surfaceBody, run, step, FP.Structural.evalExpr, evalComputeOp,
    evalOp_unfold, Option.bind, Engine.algebra, numeric, FP.Structural.bop,
    State.setReg, natLt, numericLt, numericEq, ComputeDType.eraseDType, hp]

#axiomsClean exact
end
end VeriTile.Bench.Repairs.LogsumexpFwd.PairSource
