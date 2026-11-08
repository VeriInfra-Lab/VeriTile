import bench.tritonbench_g.softmax_optimize.Proofs.Source
import bench.tritonbench_g.softmax_optimize.Proofs.Mathematical_SignedPrevious

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.Previous
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SoftmaxOptimize
set_option maxHeartbeats 4000000
noncomputable section

/-- Evaluate the actual signed cdiv expression on the helper's captured inputs. -/
theorem expression_eval (N B : Nat) (hN : 0 < N) (hB : 0 < B)
    (a b : RegName) (s : BlockState)
    (ha : s.regs .int [] a = some (Tile.scalar (N : Int)))
    (hb : s.regs .nat [] b = some (Tile.scalar B)) :
    evalOp (Source.previousExpression a b) s =
      some (Tile.scalar (Config.previous N B : Int)) := by
  have he : evalOp (Source.previousExpression a b) s =
      evalOp (FullSource.previousExpression N B) s := by
    simp only [Source.previousExpression, Op.tritonDiv, Op.truncDiv,
      FullSource.previousExpression, FullSource.signedQuotient,
      evalOp.eq_def, ha, hb, Option.bind_some, Option.bind_eq_bind, pure, Option.bind]
    apply congrArg some
    apply Tile.ext
    intro i
    change ((if decide ((((N : Int) + (B - 1)) + 0) < 0) then
        0 - (0 - (((N : Int) + (B - 1)) + 0)) / (0 + B)
      else (((N : Int) + (B - 1)) + 0) / (0 + B)) * B - B) =
      ((if decide ((((N : Int) + B - 1) + 0) < 0) then
        0 - (0 - (((N : Int) + B - 1) + 0)) / (0 + B)
      else (((N : Int) + B - 1) + 0) / (0 + B)) * B - B)
    rw [show (N : Int) + (B - 1) = (N : Int) + B - 1 by omega]
  rw [he, SignedTransport.previous_eval N B hN hB]
  rfl

def state (s : BlockState) (N B : Nat) (a b result : RegName) : BlockState :=
  (((s.setReg a .int [] (Tile.scalar (N : Int))).setReg b .nat [] (Tile.scalar B)).setReg
    result .int [] (Tile.scalar (Config.previous N B : Int))).setReg
    "prev_multiple" .int [] (Tile.scalar (Config.previous N B : Int))

theorem run (N B : Nat) (hN : 0 < N) (hB : 0 < B) (a b result : RegName)
    (hab : a ≠ b) (s : BlockState) :
    stepStmts (Source.previousCall N B a b result) s = some (state s N B a b result) := by
  unfold Source.previousCall
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := Tile.scalar (N : Int))
    (by simp only [evalOp.eq_def]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := Tile.scalar B)
    (by simp only [evalOp.eq_def]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (expression_eval N B hN hB a b _
      (by rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hab, BlockState.setReg_same])
      (by simp only [BlockState.setReg_same])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.scalar (Config.previous N B : Int))
    (by simp only [evalOp_ref, BlockState.setReg_same]))]
  rw [stepStmts.nil]
  rfl

#axiomsClean expression_eval
#axiomsClean run
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.Previous
