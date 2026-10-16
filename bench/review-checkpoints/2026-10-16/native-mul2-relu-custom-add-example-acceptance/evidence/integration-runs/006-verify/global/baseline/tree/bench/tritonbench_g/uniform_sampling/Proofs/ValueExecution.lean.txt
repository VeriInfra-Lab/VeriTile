import bench.tritonbench_g.uniform_sampling.Proofs.IndexExecution
import bench.tritonbench_g.uniform_sampling.Proofs.TupleExecution
import bench.tritonbench_g.uniform_sampling.Proofs.AffineExecution

/-! The complete first fifteen source statements actually execute. This
connects counter preparation, all five tuple captures, all four Philox words,
and the four unchanged affine statements. No random result is precomputed.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.ValueExecution
open VeriTile.Triton ScalarSpec IndexExecution CounterExecution
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

def randomState (seed offset : Int) (B : Nat) (u : BlockState) : BlockState :=
  TupleExecution.outputState seed offset (indices B u)
    (TupleExecution.capturedState seed offset (indices B u) (preparedState seed offset B u))

theorem random_step (seed offset : Int) (B : Nat) (u : BlockState)
    (hseed : Random.IntegerType.i64.inRange seed = Bool.true) :
    stepStmt (Algorithm.tupleBlock B) (preparedState seed offset B u) =
      some (randomState seed offset B u) := by
  apply TupleExecution.tuple_exec seed offset (indices B u) _ hseed
  all_goals simp [preparedState, wrappedState, indexState, counterState, lowState, boundState]

theorem random_register (seed offset : Int) (B : Nat) (u : BlockState)
    (f : Random.Direct.Field) :
    (randomState seed offset B u).regs .nat [B] (TupleExecution.fieldName f) =
      some ⟨fun i => (word seed offset (indices B u i) f).toNat⟩ := by
  cases f <;> simp [randomState, TupleExecution.outputState, TupleExecution.fieldState,
    TupleExecution.fieldName, Random.Direct.sameShapes, Random.Direct.Shapes.at,
    RandomDomain.wordField, Random.Direct.Words.at, Random.Direct.ofPointwise, word, field]
  all_goals funext i
  all_goals rfl

theorem random_prefix_exec (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (u : BlockState)
    (hseed : Random.IntegerType.i64.inRange seed = Bool.true) :
    stepStmts ((Algorithm.kernel swapped y N seed offset lower upper B).body.take 11) u =
      some (randomState seed offset B u) := by
  change stepStmts
    (((Algorithm.kernel swapped y N seed offset lower upper B).body.take 10) ++
      [Algorithm.tupleBlock B]) u = _
  rw [steps_append, IndexExecution.prefix_exec, Option.bind_some]
  simp only [stepStmts, random_step seed offset B u hseed]

def affineAssignState {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (lower upper : ℝ) (f : Random.Direct.Field) (s : BlockState) : BlockState :=
  s.setReg (TupleExecution.fieldName f) .real [B]
    (AffineExecution.affineValues (fun i => word seed offset (i4 i) f) lower upper)

def affineState {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (lower upper : ℝ) (s : BlockState) : BlockState :=
  affineAssignState seed offset i4 lower upper .c3
    (affineAssignState seed offset i4 lower upper .c2
      (affineAssignState seed offset i4 lower upper .c1
        (affineAssignState seed offset i4 lower upper .c0 s)))

theorem four_steps {a b c d : Stmt} {s0 s1 s2 s3 s4 : BlockState}
    (ha : stepStmt a s0 = some s1) (hb : stepStmt b s1 = some s2)
    (hc : stepStmt c s2 = some s3) (hd : stepStmt d s3 = some s4) :
    stepStmts [a, b, c, d] s0 = some s4 := by
  simp only [stepStmts, ha, hb, hc, hd]

theorem affines_exec {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (lower upper : ℝ) (s : BlockState)
    (hw : ∀ f : Random.Direct.Field, s.regs .nat [B] (TupleExecution.fieldName f) =
      some ⟨fun i => (word seed offset (i4 i) f).toNat⟩) :
    stepStmts [Algorithm.affineStmt B "r0" lower upper,
      Algorithm.affineStmt B "r1" lower upper, Algorithm.affineStmt B "r2" lower upper,
      Algorithm.affineStmt B "r3" lower upper] s = some (affineState seed offset i4 lower upper s) := by
  let t0 := affineAssignState seed offset i4 lower upper .c0 s
  let t1 := affineAssignState seed offset i4 lower upper .c1 t0
  let t2 := affineAssignState seed offset i4 lower upper .c2 t1
  let t3 := affineAssignState seed offset i4 lower upper .c3 t2
  have h0 : stepStmt (Algorithm.affineStmt B "r0" lower upper) s = some t0 :=
    AffineExecution.affine_step _ "r0" lower upper s (hw .c0)
  have h1 : stepStmt (Algorithm.affineStmt B "r1" lower upper) t0 = some t1 := by
    apply AffineExecution.affine_step
    simpa [t0, affineAssignState, TupleExecution.fieldName] using hw .c1
  have h2 : stepStmt (Algorithm.affineStmt B "r2" lower upper) t1 = some t2 := by
    apply AffineExecution.affine_step
    simpa [t1, t0, affineAssignState, TupleExecution.fieldName] using hw .c2
  have h3 : stepStmt (Algorithm.affineStmt B "r3" lower upper) t2 = some t3 := by
    apply AffineExecution.affine_step
    simpa [t2, t1, t0, affineAssignState, TupleExecution.fieldName] using hw .c3
  exact four_steps h0 h1 h2 h3

def valueState (seed offset : Int) (lower upper : ℝ) (B : Nat) (u : BlockState) : BlockState :=
  affineState seed offset (indices B u) lower upper (randomState seed offset B u)

theorem value_register (seed offset : Int) (lower upper : ℝ) (B : Nat) (u : BlockState)
    (f : Random.Direct.Field) :
    (valueState seed offset lower upper B u).regs .real [B] (TupleExecution.fieldName f) =
      some ⟨fun i => some (value seed offset (indices B u i) f lower upper)⟩ := by
  cases f <;> simp [valueState, affineState, affineAssignState, TupleExecution.fieldName,
    AffineExecution.affineValues, value]

theorem affine_phase_exec (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (u : BlockState) :
    stepStmts (((Algorithm.kernel swapped y N seed offset lower upper B).body.drop 11).take 4)
      (randomState seed offset B u) = some (valueState seed offset lower upper B u) := by
  exact affines_exec seed offset (indices B u) lower upper _ (random_register seed offset B u)

theorem prefix_exec (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (u : BlockState)
    (hseed : Random.IntegerType.i64.inRange seed = Bool.true) :
    stepStmts ((Algorithm.kernel swapped y N seed offset lower upper B).body.take 15) u =
      some (valueState seed offset lower upper B u) := by
  change stepStmts
    (((Algorithm.kernel swapped y N seed offset lower upper B).body.take 11) ++
      ((Algorithm.kernel swapped y N seed offset lower upper B).body.drop 11).take 4) u = _
  rw [steps_append, random_prefix_exec swapped y N seed offset lower upper B u hseed,
    Option.bind_some, affine_phase_exec]

theorem affineState_frame {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (lower upper : ℝ) (s : BlockState) :
    (affineState seed offset i4 lower upper s).mem = s.mem ∧
    (affineState seed offset i4 lower upper s).pids = s.pids ∧
    (affineState seed offset i4 lower upper s).undef = s.undef ∧
    (affineState seed offset i4 lower upper s).numPids = s.numPids :=
  ⟨rfl, rfl, rfl, rfl⟩

theorem valueState_frame (seed offset : Int) (lower upper : ℝ) (B : Nat) (u : BlockState) :
    (valueState seed offset lower upper B u).mem = u.mem ∧
    (valueState seed offset lower upper B u).pids = u.pids ∧
    (valueState seed offset lower upper B u).undef = u.undef ∧
    (valueState seed offset lower upper B u).numPids = u.numPids := by
  have ha := affineState_frame seed offset (indices B u) lower upper (randomState seed offset B u)
  have hr := TupleExecution.outputState_frame seed offset (indices B u) (preparedState seed offset B u)
  have hp := preparedState_frame seed offset B u
  exact ⟨ha.1.trans (hr.1.trans hp.1),
    ha.2.1.trans (hr.2.1.trans hp.2.1),
    ha.2.2.1.trans (hr.2.2.1.trans hp.2.2.1),
    ha.2.2.2.trans (hr.2.2.2.trans hp.2.2.2)⟩

#axiomsClean random_step
#axiomsClean random_register
#axiomsClean random_prefix_exec
#axiomsClean four_steps
#axiomsClean affines_exec
#axiomsClean value_register
#axiomsClean affine_phase_exec
#axiomsClean prefix_exec
#axiomsClean affineState_frame
#axiomsClean valueState_frame

end
end VeriTile.Bench.TritonBenchG.UniformSampling.ValueExecution
