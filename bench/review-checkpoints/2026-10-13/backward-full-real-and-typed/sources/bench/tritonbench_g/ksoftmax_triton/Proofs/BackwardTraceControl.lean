import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardStaticControl
import VeriTile.Triton.Memory.TraceRewrite

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.StaticControl
open VeriTile.Triton
set_option maxHeartbeats 800000
set_option maxRecDepth 16000

/-- Append safety follows the actual intermediate state, including failure. -/
theorem trace_append (bounds : RegionBounds) (left right : List Stmt) (s : BlockState) :
    Stmt.TraceSafeList bounds (left ++ right) s ↔
      Stmt.TraceSafeList bounds left s ∧
        (∀ t, stepStmts left s = some t → Stmt.TraceSafeList bounds right t) := by
  exact TraceRewrite.append_iff bounds left right s

theorem trace_if_true (bounds : RegionBounds) (body rest : List Stmt) (s : BlockState) :
    Stmt.TraceSafeList bounds (.ifThen (.constBool Bool.true) body :: rest) s ↔
      Stmt.TraceSafeList bounds (body ++ rest) s := by
  rw [Stmt.TraceSafeList, trace_append]
  cases h : stepStmts body s <;>
    simp [Stmt.TraceSafe, Op.SafeAt.eq_def, evalOp.eq_def, if_true, h]

theorem trace_if_false (bounds : RegionBounds) (body rest : List Stmt) (s : BlockState) :
    Stmt.TraceSafeList bounds (.ifThen (.constBool Bool.false) body :: rest) s ↔
      Stmt.TraceSafeList bounds rest s := by
  rw [Stmt.TraceSafeList]
  simp [Stmt.TraceSafe, Op.SafeAt.eq_def, evalOp.eq_def, if_false]

theorem trace_ifElse_true (bounds : RegionBounds) (left right rest : List Stmt) (s : BlockState) :
    Stmt.TraceSafeList bounds (.ifThenElse (.constBool Bool.true) left right :: rest) s ↔
      Stmt.TraceSafeList bounds (left ++ rest) s := by
  rw [Stmt.TraceSafeList, trace_append]
  cases h : stepStmts left s <;>
    simp [Stmt.TraceSafe, Op.SafeAt.eq_def, evalOp.eq_def, ifElse_true, h]

theorem trace_ifElse_false (bounds : RegionBounds) (left right rest : List Stmt) (s : BlockState) :
    Stmt.TraceSafeList bounds (.ifThenElse (.constBool Bool.false) left right :: rest) s ↔
      Stmt.TraceSafeList bounds (right ++ rest) s := by
  rw [Stmt.TraceSafeList, trace_append]
  cases h : stepStmts right s <;>
    simp [Stmt.TraceSafe, Op.SafeAt.eq_def, evalOp.eq_def, ifElse_false, h]

theorem trace_assign_real_cast (bounds : RegionBounds) (shape : TileShape)
    (name : RegName) (e : Op .real shape) (rest : List Stmt) (s : BlockState) :
    Stmt.TraceSafeList bounds (.assign .real shape name (.castFloat .real .real e) :: rest) s ↔
      Stmt.TraceSafeList bounds (.assign .real shape name e :: rest) s := by
  rw [Stmt.TraceSafeList, Stmt.TraceSafeList, assign_real_cast]
  have hsafe : (.castFloat .real .real e : Op .real shape).SafeAt bounds s = e.SafeAt bounds s := by
    conv_lhs => rw [Op.SafeAt.eq_def]
    rfl
  simp only [Stmt.TraceSafe]
  exact and_congr (Iff.of_eq hsafe) Iff.rfl

#axiomsClean trace_append
#axiomsClean trace_if_true
#axiomsClean trace_if_false
#axiomsClean trace_ifElse_true
#axiomsClean trace_ifElse_false
#axiomsClean trace_assign_real_cast
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.StaticControl
