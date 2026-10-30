import bench.tritonbench_g.logsumexp_fwd.Proofs.RawKernel
import VeriTile.Triton.Float.StructuralSafety

/-! Access bounds follow the quoted source's signed row/block arithmetic,
actual masks and runtime scale binding. The same proof covers Real and all
selected native reduction engines. -/
namespace VeriTile.Bench.Repairs.LogsumexpFwd
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem block_traceSafe [Inhabited A] (engine : Engine A) (x z : RegionName)
    (D B : Nat) (hasScale : Bool) (scale : A) (s : State A)
    (bounds : RegionName → Nat) (hB : 0 < B)
    (parameter : s.regs .real [] "scale" = some (fun _ => scale))
    (input : ∀ i : Fin B, s.pids 1 * B + i.val < D →
      Source.inputAddress (s.pids 0) (s.pids 1) D B i.val < bounds x)
    (output : Source.outputAddress (s.pids 0) (s.pids 1) D B < bounds z) :
    FP.Structural.TraceSafe engine.algebra bounds (rowBlock x z D B hasScale) s := by
  have hsum : 1 ≤ D + B := by omega
  cases hasScale
  all_goals simp (maxSteps := 1800000) [FP.Structural.TraceSafe, rowBlock, ComputeKernel.surfaceBody,
    Safety.traceSafeList, Safety.stepSafe, Safety.safeExpr, Safety.safeCompute,
    Safety.safeOp.eq_def, Safety.safeAddress, Safety.safeMask, Safety.addressSafe,
    Safety.maskActive, run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    store, Option.bind, ComputeDType.eraseDType, numeric, bop, natLt, numericLt,
    numericEq, Region.cast, Engine.algebra, State.setReg, parameter, TileShape.axisDim, hB]
  all_goals simp only [← Int.natCast_one, ← Int.natCast_sub hsum, ← Int.natCast_add,
    ← Int.natCast_mul, ← Int.natCast_ediv, Int.toNat_natCast, Int.ofNat_lt]
  all_goals exact ⟨input, output⟩

#axiomsClean block_traceSafe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LogsumexpFwd
