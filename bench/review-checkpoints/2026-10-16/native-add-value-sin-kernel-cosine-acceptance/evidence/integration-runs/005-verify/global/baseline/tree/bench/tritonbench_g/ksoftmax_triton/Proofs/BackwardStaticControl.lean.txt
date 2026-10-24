import VeriTile.Triton
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.StaticControl
open VeriTile.Triton

/-! Proof-only equations for the original static branches. They preserve
all reached statements, including failure, and do not replace a source body. -/

theorem if_true (body : List Stmt) (s : BlockState) :
    stepStmt (.ifThen (.constBool Bool.true) body) s = stepStmts body s := by
  simp [stepStmt, evalOp.eq_def]

theorem if_false (body : List Stmt) (s : BlockState) :
    stepStmt (.ifThen (.constBool Bool.false) body) s = some s := by
  simp [stepStmt, evalOp.eq_def]

theorem ifElse_true (left right : List Stmt) (s : BlockState) :
    stepStmt (.ifThenElse (.constBool Bool.true) left right) s = stepStmts left s := by
  simp [stepStmt, evalOp.eq_def]

theorem ifElse_false (left right : List Stmt) (s : BlockState) :
    stepStmt (.ifThenElse (.constBool Bool.false) left right) s = stepStmts right s := by
  simp [stepStmt, evalOp.eq_def]

theorem cons_bind (st : Stmt) (rest : List Stmt) (s : BlockState) :
    stepStmts (st :: rest) s = (stepStmt st s).bind (stepStmts rest) := by
  rw [stepStmts]
  cases stepStmt st s <;> rfl

theorem assign_real_cast (shape : TileShape) (name : RegName)
    (e : Op .real shape) (s : BlockState) :
    stepStmt (.assign .real shape name (.castFloat .real .real e)) s =
      stepStmt (.assign .real shape name e) s := by
  have hc : evalOp (.castFloat .real .real e) s = evalOp e s := by
    calc
      evalOp (.castFloat .real .real e) s = (do
        let value ← evalOp e s
        some (⟨fun i => FloatDType.real.cast .real (value.data i)⟩ : Tile .real shape)) :=
        evalOp_castFloat .real .real e s
      _ = evalOp e s := by cases evalOp e s <;> rfl
  simp only [stepStmt]
  exact congrArg (fun v : Option (Tile .real shape) =>
    v.bind (fun value => some (s.setReg name .real shape value))) hc

#axiomsClean if_true
#axiomsClean if_false
#axiomsClean ifElse_true
#axiomsClean ifElse_false
#axiomsClean cons_bind
#axiomsClean assign_real_cast
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.StaticControl
