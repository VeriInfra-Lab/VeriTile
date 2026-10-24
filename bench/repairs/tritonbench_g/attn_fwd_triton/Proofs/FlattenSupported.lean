import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.Execution
import VeriTile.Triton.Memory.Flatten

/-! Both complete repaired sources use the supported flat-memory syntax. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.FlattenSupported
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttnFwdCausal
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem supported (p : Parameters) (BM BN BD : Nat) (reciprocal : Bool) :
    StmtList.FlattenOk (Verified.code BM BN BD p reciprocal) := by
  rw [Execution.body,Coordinates.source,Pointers.source,Initialization.source]
  cases reciprocal <;> simp [Coordinates.code,Coordinates.ceilExpr,
    Pointers.code,Pointers.rowExpr,Pointers.keyExpr,RealPointers.rowExpr,RealPointers.queryScaleExpr,RealPointers.keyScaleExpr,
    Initialization.code,RealInit.code,queryMask,
    TwoLoops.code,FirstSetup.code,FirstSetup.copies,RealFirstSetup.copies,
    Setup.code,Setup.lowerExpr,Setup.upperExpr,RealSetup.lowerExpr,RealSetup.upperExpr,
    loop,sourceIteration,flatIteration,loadPrefix,keyMask,valueMask,tail,guard,
    RealBody.score,RealBody.scoreExpr,RealBody.center,RealBody.centerExpr,RealBody.probability,
    suffix,RealBody.rescale,valueLoad,RealBody.accumulate,advance,
    epilogue,RealSource.normalization,RealExpressions.maskExpr,
    StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.FlattenSupported
