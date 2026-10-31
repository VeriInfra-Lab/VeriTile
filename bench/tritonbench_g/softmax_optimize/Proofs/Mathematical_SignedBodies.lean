import VeriTile.Triton.Semantics.ProgramRewrite
import bench.tritonbench_g.softmax_optimize.Proofs.Mathematical_SourceShape
import bench.tritonbench_g.softmax_optimize.Proofs.Mathematical_LegacyReal

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
open VeriTile.Triton FullSource
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

def outputPointer (Y : RegionName) (B : Nat) : Stmt :=
  .assign .ptr [B] "output_ptrs"
    (.ptrAdd .scalarL (.ptrBase Y) (.castIntToNat (.ref .int [B] "offset")))

/-- The dtype-erased source retains its mathematical maximum as max2. -/
def sourceMaximum (B : Nat) : Stmt :=
  .assign .real [B] "new_m"
    (.max2 (.consSame .nil) (.ref .real [B] "m") (.ref .real [B] "inp"))

private theorem maximum_statement_step (B : Nat) (rest : List Stmt) (s : BlockState) :
    stepStmts (sourceMaximum B :: rest) s =
    stepStmts (.assign .real [B] "new_m"
      (.maximumFloat .real (.consSame .nil) (.ref .real [B] "m") (.ref .real [B] "inp")) :: rest) s := by
  simp only [sourceMaximum, stepStmts, stepStmt]
  erw [evalOp_maximumFloat_real]

private theorem maximum_after_prefix_step (B : Nat) (pre rest : List Stmt) (s : BlockState) :
    stepStmts (pre ++ sourceMaximum B :: rest) s =
    stepStmts (pre ++ .assign .real [B] "new_m"
      (.maximumFloat .real (.consSame .nil) (.ref .real [B] "m") (.ref .real [B] "inp")) :: rest) s := by
  rw [ProgramRewrite.append, ProgramRewrite.append]
  congr 1
  funext t
  exact maximum_statement_step B rest t

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
