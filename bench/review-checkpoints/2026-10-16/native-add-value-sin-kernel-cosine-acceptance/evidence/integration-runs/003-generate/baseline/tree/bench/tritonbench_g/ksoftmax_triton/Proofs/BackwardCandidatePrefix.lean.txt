import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardSource
import VeriTile.Triton.Semantics.ProgramRewrite
import VeriTile.Triton.Memory.TraceRewrite
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.Candidate
open VeriTile.Triton
set_option maxHeartbeats 800000
set_option maxRecDepth 16000

/-- The first four assignments, used only to prove a decomposition of the
actual source. All loads, control flow and stores remain in its exact suffix. -/
def indexPrefix (v : Bool) (GradOut : RegionName) (gm gn D : Nat) : List Stmt :=
  [.assign .nat [] "m" (.programId 0),
   .assign .nat [] "n" (.programId 1),
   .assign .nat [D] "k" (.arange D),
   .assign .ptr [D] "grad_out_ptrs"
     (.ptrAdd .scalarL (.ptrBase GradOut)
       (.add .nat .scalarL
         (.add .nat .nil
           (if v then .mul .nat .nil (.constNat gm) (.ref .nat [] "m")
            else .mul .nat .nil (.ref .nat [] "m") (.constNat gm))
           (.mul .nat .nil (.ref .nat [] "n") (.constNat gn)))
         (.ref .nat [D] "k")))]

/-- Address commutation preserves every flat prefix state, even when the
initial registers or memory are arbitrary. No erasure/FP theorem is assumed. -/
theorem prefix_flat (A : FlatAlloc) (GradOut : RegionName) (gm gn D : Nat)
    (s : BlockState) :
    stepStmts (A.flattenStmts (indexPrefix Bool.true GradOut gm gn D)) s =
      stepStmts (A.flattenStmts (indexPrefix Bool.false GradOut gm gn D)) s := by
  simp [indexPrefix, FlatAlloc.flattenStmts, FlatAlloc.flattenStmt, FlatAlloc.flattenOp,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.ptrAdd, NumericDType.mul,
    Nat.mul_comm]

/-- A narrow source-link check before lifting the same exact decomposition
to independently chosen storage types. -/
theorem fp16_body (v : Bool) (GradIn GradOut Out : RegionName)
    (bm bn gm gn om on K D : Nat) (log causal fp16 : Bool) :
    (Typed.homogeneousKernel .fp16 v GradIn GradOut Out
      bm bn gm gn om on K D log causal fp16).eraseDType.toAlgKernel.body =
    indexPrefix v GradOut gm gn D ++
      (Typed.homogeneousKernel .fp16 Bool.false GradIn GradOut Out
        bm bn gm gn om on K D log causal fp16).eraseDType.toAlgKernel.body.drop 4 := by
  have hp (w : Bool) :
      (Typed.homogeneousKernel .fp16 w GradIn GradOut Out
        bm bn gm gn om on K D log causal fp16).toAlgorithm? =
      .ok (Typed.homogeneousKernel .fp16 w GradIn GradOut Out
        bm bn gm gn om on K D log causal fp16).toAlgKernel := by
    cases w <;> rfl
  simp only [ComputeKernel.eraseDType, hp, ComputeKernel.toAlgKernel_fromKernelBody]
  cases v
  all_goals
    dsimp only [Typed.homogeneousKernel, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Except.bind, Except.pure, Bind.bind, Pure.pure, Kernel.eraseDType,
      ComputeDType.eraseDType]
    simp only [Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
      MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType,
      NumericDType.eraseDType, ComparableDType.eraseDType, FloatDType.toTileDType,
      Region.cast, indexPrefix, List.drop_succ_cons, List.drop_zero]
    rfl

#axiomsClean prefix_flat
#axiomsClean fp16_body
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.Candidate
