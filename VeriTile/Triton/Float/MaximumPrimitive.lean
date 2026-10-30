import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Semantics.EvalOp
import VeriTile.Triton.Float.EvalOpR
import VeriTile.Meta.StatementAudit

namespace VeriTile.Triton

/-- The old mathematical maximum is the real projection of the floating
primitive. This lemma is deliberately about the mathematical interpreter;
it imposes no comparison/select law on an arbitrary floating backend. -/
theorem evalOp_maximumFloat_real (bc : Broadcast a b out)
    (x : Op .real a) (y : Op .real b) (s : BlockState) :
    evalOp (.maximumFloat .real bc x y) s = evalOp (.max2 bc x y) s := by
  simp only [evalOp, FloatDType.toWithBot, FloatDType.ofWithBot]
  rfl

/-- All physical floating formats project to the same real maximum, retaining
both operand projections and their broadcasting. -/
theorem eraseDType_maximumFloat (dtype : FloatDType) (bc : Broadcast a b out)
    (x : Op dtype.toTileDType a) (y : Op dtype.toTileDType b) :
    HEq (Op.maximumFloat dtype bc x y).eraseDType
      (Op.max2 bc
        (cast (congrArg (fun d => Op d a) (eraseDType_float dtype)) x.eraseDType)
        (cast (congrArg (fun d => Op d b) (eraseDType_float dtype)) y.eraseDType)) := by
  cases dtype <;> rw [Op.eraseDType.eq_def] <;> rfl

/-- Compatibility with the former mathematical comparison/select lowering.
This is only a theorem about WithBot Real, never a law for IEEE NaN payloads. -/
theorem evalOp_maximumFloat_real_select (bc : Broadcast a b out)
    (x : Op .real a) (y : Op .real b) (s : BlockState) :
    evalOp (.maximumFloat .real bc x y) s =
      evalOp (.where (.gt .real bc x y)
        (.remap out (Broadcast.leftIndex bc) x)
        (.remap out (Broadcast.rightIndex bc) y)) s := by
  rw [evalOp_maximumFloat_real]
  cases hx : evalOp x s <;> cases hy : evalOp y s <;>
    simp [evalOp, hx, hy, Tile.bop, Tile.cop, Tile.select, Tile.remap, ComparableDType.gt, max_def, ← not_le, ite_not]


/-- The rounding-model interpreter has the same mathematical max projection;
its operand evaluation still uses the supplied rounding model. -/
theorem evalOpR_maximumFloat_real (R : RoundingModel) (bc : Broadcast a b out)
    (x : Op .real a) (y : Op .real b) (s : BlockState) :
    evalOpR R (.maximumFloat .real bc x y) s = evalOpR R (.max2 bc x y) s := by
  simp only [evalOpR, FloatDType.toWithBot, FloatDType.ofWithBot]
  rfl

/-- Compatibility with the former mathematical comparison/select lowering.
This is only a theorem about WithBot Real, never a law for IEEE NaN payloads. -/
theorem evalOpR_maximumFloat_real_select (R : RoundingModel) (bc : Broadcast a b out)
    (x : Op .real a) (y : Op .real b) (s : BlockState) :
    evalOpR R (.maximumFloat .real bc x y) s =
      evalOpR R (.where (.gt .real bc x y)
        (.remap out (Broadcast.leftIndex bc) x)
        (.remap out (Broadcast.rightIndex bc) y)) s := by
  rw [evalOpR_maximumFloat_real]
  cases hx : evalOpR R x s <;> cases hy : evalOpR R y s <;>
    simp [evalOpR, hx, hy, Tile.bop, Tile.cop, Tile.select, Tile.remap, ComparableDType.gt, max_def, ← not_le, ite_not]


namespace FP.Structural
/-- The floating interpretation passes the actual dtype and precision to an
independent maximum callback. Neither ordering nor finiteness is a premise. -/
theorem evalOp_maximumFloat {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (dtype : FloatDType) (bc : Broadcast a b out)
    (x : Op dtype.toTileDType a) (y : Op dtype.toTileDType b) (s : State α) :
    evalOp M p (.maximumFloat dtype bc x y) s = (do
      let vx ← evalOp M p x s
      let vy ← evalOp M p y s
      pure (bop (fun u v => ofFloat dtype (M.binary p dtype .max
        (toFloat dtype u) (toFloat dtype v))) bc vx vy)) := by
  rw [evalOp_unfold]
end FP.Structural

#axiomsClean evalOp_maximumFloat_real
#axiomsClean eraseDType_maximumFloat
#axiomsClean evalOp_maximumFloat_real_select
#axiomsClean evalOpR_maximumFloat_real
#axiomsClean evalOpR_maximumFloat_real_select
#axiomsClean FP.Structural.evalOp_maximumFloat
end VeriTile.Triton
