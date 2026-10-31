import bench.tritonbench_g.uniform_sampling.Proofs.Mathematical_Algorithm
import bench.tritonbench_g.uniform_sampling.Proofs.Mathematical_CounterOperands
import VeriTile.Triton.Semantics.Eval

/-! Actual execution of the source's materializations, signed64 casts and
two masked counter words. The local mathematical evaluator accepts arbitrary
integers; the complete source contract separately retains both original i64
ABI conditions. No execution result is assumed as an input specification.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.CounterExecution
open VeriTile.Triton ScalarSpec
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

def boundState (seed offset : Int) (B : Nat) (u : BlockState) : BlockState :=
  let s := u.setReg "philox_seed_arg" .int [] (Tile.scalar seed)
  let s := s.setReg "philox_offset_arg" .int [] (Tile.scalar offset)
  let s := s.setReg "BLOCK_arg" .nat [] (Tile.scalar B)
  let s := s.setReg "philox_seed" .int [] (Tile.scalar seed)
  s.setReg "philox_offset" .int [] (Tile.scalar offset)

def lowState (seed offset : Int) (B : Nat) (u : BlockState) : BlockState :=
  (boundState seed offset B u).setReg "c0" .nat []
    (Tile.scalar (BitVec.ofInt 32 offset).toNat)

def counterState (seed offset : Int) (B : Nat) (u : BlockState) : BlockState :=
  (lowState seed offset B u).setReg "c1" .nat [] (Tile.scalar (counterHigh offset).toNat)

theorem low_eval (seed offset : Int) (B : Nat) (u : BlockState) :
    evalOp (.intToUnsigned 32 (.intBitAnd 64 .nil
      (.ref .int [] "philox_offset") (.constInt 0xffffffff))) (boundState seed offset B u) =
      some (Tile.scalar (BitVec.ofInt 32 offset).toNat) := by
  apply CounterOperands.eval_low_mask
  simp [evalOp.eq_def, boundState]

theorem high_eval (seed offset : Int) (B : Nat) (u : BlockState) :
    evalOp (.intToUnsigned 32 (.intBitAnd 64 .nil
      (.intShiftRight .nil (.ref .int [] "philox_offset") (.constNat 32))
      (.constInt 0xffffffff))) (lowState seed offset B u) =
      some (Tile.scalar (counterHigh offset).toNat) := by
  apply CounterOperands.eval_high_mask
  simp [evalOp.eq_def, lowState, boundState]

theorem steps_append (before after : List Stmt) (s : BlockState) :
    stepStmts (before ++ after) s = (stepStmts before s).bind (stepStmts after) := by
  induction before generalizing s with
  | nil => simp only [List.nil_append, stepStmts.nil, Option.bind_some]
  | cons head rest ih =>
      simp only [List.cons_append, stepStmts]
      cases stepStmt head s with
      | none => rfl
      | some next => exact ih next

theorem counterState_frame (seed offset : Int) (B : Nat) (u : BlockState) :
    (counterState seed offset B u).mem = u.mem ∧
    (counterState seed offset B u).pids = u.pids ∧
    (counterState seed offset B u).undef = u.undef ∧
    (counterState seed offset B u).numPids = u.numPids :=
  ⟨rfl, rfl, rfl, rfl⟩

#axiomsClean low_eval
#axiomsClean high_eval
#axiomsClean counterState_frame

end
end VeriTile.Bench.TritonBenchG.UniformSampling.CounterExecution
