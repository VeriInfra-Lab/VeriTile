import bench.tritonbench_g.uniform_sampling.Proofs.CounterExecution

/-! Actual execution of the remaining three counter-preparation statements.
The low-word addition is a genuine uint32 wrap, including when the incoming
offset is negative. The high word receives no carry from this addition.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.IndexExecution
open VeriTile.Triton ScalarSpec CounterExecution
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

def indices (B : Nat) (u : BlockState) : TileIndex [B] → Nat :=
  fun i => SourceDomain.counterIndex (u.pids 0) B i.1.val

def indexState (seed offset : Int) (B : Nat) (u : BlockState) : BlockState :=
  (counterState seed offset B u).setReg "i4" .nat [B] ⟨indices B u⟩

def wrappedState (seed offset : Int) (B : Nat) (u : BlockState) : BlockState :=
  (indexState seed offset B u).setReg "c0" .nat [B]
    ⟨fun i => (counterLow offset (indices B u i)).toNat⟩

def preparedState (seed offset : Int) (B : Nat) (u : BlockState) : BlockState :=
  (wrappedState seed offset B u).setReg "_O" .nat [B] ⟨fun _ => 0⟩

theorem index_eval (swapped : Bool) (seed offset : Int) (B : Nat) (u : BlockState) :
    evalOp (Algorithm.indexOp swapped B) (counterState seed offset B u) =
      some ⟨indices B u⟩ := by
  cases swapped <;>
    simp [-Tile.scalar_eta, Algorithm.indexOp, evalOp.eq_def, counterState, lowState, boundState,
      Tile.bop, Tile.scalar, Tile.vec, NumericDType.add, NumericDType.mul,
      Nat.mul_comm]
  all_goals funext i
  all_goals simp [indices, SourceDomain.counterIndex, Nat.mul_comm]

theorem wrapped_eval (seed offset : Int) (B : Nat) (u : BlockState) :
    evalOp (.intToUnsigned 32 (.add .int .scalarL
      (.castNatToInt (.ref .nat [] "c0")) (.castNatToInt (.ref .nat [B] "i4"))))
      (indexState seed offset B u) =
        some ⟨fun i => (counterLow offset (indices B u i)).toNat⟩ := by
  have hl : (indexState seed offset B u).regs .nat [] "c0" =
      some (Tile.scalar (BitVec.ofInt 32 offset).toNat) := by
    simp [indexState, counterState, lowState]
  have hi : (indexState seed offset B u).regs .nat [B] "i4" = some ⟨indices B u⟩ := by
    simp [indexState]
  rw [eval_intToUnsigned]
  simp only [evalOp.eq_def, hl, hi, bind, Option.bind, pure, Tile.bop, Tile.scalar,
    NumericDType.add, Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL]
  exact congrArg (fun f : TileIndex [B] → Nat => some (⟨f⟩ : Tile .nat [B]))
    (funext fun i => CounterOperands.wrapped_low_add offset (indices B u i))

theorem zeros_eval (seed offset : Int) (B : Nat) (u : BlockState) :
    evalOp (.intToUnsigned 32 (.mul .int .scalarR
      (.castNatToInt (.ref .nat [B] "c0")) (.constInt 0)))
      (wrappedState seed offset B u) = some ⟨fun _ => 0⟩ := by
  rw [eval_intToUnsigned]
  simp [-Tile.scalar_eta, evalOp.eq_def, wrappedState, Tile.bop, Tile.scalar, NumericDType.mul]

theorem assign_step_of_eval {dtype : TileDType} {shape : TileShape}
    (name : RegName) (e : Op dtype shape) (s : BlockState) (v : Tile dtype shape)
    (h : evalOp e s = some v) :
    stepStmt (.assign dtype shape name e) s = some (s.setReg name dtype shape v) := by
  simp only [stepStmt]
  rw [h]
  rfl

theorem three_steps {a b c : Stmt} {s0 s1 s2 s3 : BlockState}
    (ha : stepStmt a s0 = some s1) (hb : stepStmt b s1 = some s2)
    (hc : stepStmt c s2 = some s3) :
    stepStmts [a, b, c] s0 = some s3 := by
  simp only [stepStmts, ha, hb, hc]

theorem preparation_exec (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (u : BlockState) :
    stepStmts (((Algorithm.kernel swapped y N seed offset lower upper B).body.drop 7).take 3)
      (counterState seed offset B u) = some (preparedState seed offset B u) := by
  change stepStmts [
    .assign .nat [B] "i4" (Algorithm.indexOp swapped B),
    .assign .nat [B] "c0" (.intToUnsigned 32 (.add .int .scalarL
      (.castNatToInt (.ref .nat [] "c0")) (.castNatToInt (.ref .nat [B] "i4")))),
    .assign .nat [B] "_O" (.intToUnsigned 32 (.mul .int .scalarR
      (.castNatToInt (.ref .nat [B] "c0")) (.constInt 0)))]
      (counterState seed offset B u) = some (preparedState seed offset B u)
  exact three_steps
    (assign_step_of_eval "i4" _ _ _ (index_eval swapped seed offset B u))
    (assign_step_of_eval "c0" _ _ _ (wrapped_eval seed offset B u))
    (assign_step_of_eval "_O" _ _ _ (zeros_eval seed offset B u))

theorem prefix_exec (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (u : BlockState) :
    stepStmts ((Algorithm.kernel swapped y N seed offset lower upper B).body.take 10) u =
      some (preparedState seed offset B u) := by
  change stepStmts
    (((Algorithm.kernel swapped y N seed offset lower upper B).body.take 7) ++
      ((Algorithm.kernel swapped y N seed offset lower upper B).body.drop 7).take 3) u = _
  rw [steps_append, CounterExecution.prefix_exec, Option.bind_some, preparation_exec]

theorem preparedState_frame (seed offset : Int) (B : Nat) (u : BlockState) :
    (preparedState seed offset B u).mem = u.mem ∧
    (preparedState seed offset B u).pids = u.pids ∧
    (preparedState seed offset B u).undef = u.undef ∧
    (preparedState seed offset B u).numPids = u.numPids :=
  ⟨rfl, rfl, rfl, rfl⟩

#axiomsClean index_eval
#axiomsClean wrapped_eval
#axiomsClean zeros_eval
#axiomsClean assign_step_of_eval
#axiomsClean three_steps
#axiomsClean preparation_exec
#axiomsClean prefix_exec
#axiomsClean preparedState_frame

end
end VeriTile.Bench.TritonBenchG.UniformSampling.IndexExecution
