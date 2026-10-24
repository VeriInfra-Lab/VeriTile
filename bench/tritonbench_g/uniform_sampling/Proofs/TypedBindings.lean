import bench.tritonbench_g.uniform_sampling.Proofs.TypedCounterOperands
import bench.tritonbench_g.uniform_sampling.Proofs.Erasure

/-! Successful typed execution of the original argument materializations and
signed64 counter split. The checked source equality preserves the actual
compute-facing statements; no erased execution is substituted for this run.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.TypedBindings
open VeriTile.Triton ScalarSpec TypedSource FP.Structural
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 10000

def body (seed offset : Int) (B : Nat) : List ComputeStmt := [
  .assign .int [] "philox_seed_arg" (.alg (.constInt seed)),
  .assign .int [] "philox_offset_arg" (.alg (.constInt offset)),
  .assign .nat [] "BLOCK_arg" (.alg (.constNat B)),
  .assign .int [] "philox_seed" (.alg (.ref .int [] "philox_seed_arg")),
  .assign .int [] "philox_offset" (.alg (.ref .int [] "philox_offset_arg")),
  .assign .nat [] "c0" (.alg (.intToUnsigned 32
    (.intBitAnd 64 .nil (.ref .int [] "philox_offset") (.constInt 0xffffffff)))),
  .assign .nat [] "c1" (.alg (.intToUnsigned 32
    (.intBitAnd 64 .nil (.intShiftRight .nil (.ref .int [] "philox_offset") (.constNat 32))
      (.constInt 0xffffffff))))]

theorem source_prefix (storage : Storage) (swapped : Bool) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : ℝ) (B : Nat)
    (hB : SourceDomain.LegalBlock B) :
    (source storage swapped y N seed offset lower upper B hB).surfaceBody.take 7 =
      body seed offset B := by
  cases storage <;> cases swapped <;> rfl

def boundState {α : Type} (seed offset : Int) (B : Nat) (u : State α) : State α :=
  let s := u.setReg "philox_seed_arg" .int [] (fun _ => seed)
  let s := s.setReg "philox_offset_arg" .int [] (fun _ => offset)
  let s := s.setReg "BLOCK_arg" .nat [] (fun _ => B)
  let s := s.setReg "philox_seed" .int [] (fun _ => seed)
  s.setReg "philox_offset" .int [] (fun _ => offset)

def lowState {α : Type} (seed offset : Int) (B : Nat) (u : State α) : State α :=
  (boundState seed offset B u).setReg "c0" .nat []
    (fun _ => (BitVec.ofInt 32 offset).toNat)

def counterState {α : Type} (seed offset : Int) (B : Nat) (u : State α) : State α :=
  (lowState seed offset B u).setReg "c1" .nat [] (fun _ => (counterHigh offset).toNat)

theorem bindings_exec {α : Type} [Inhabited α] (M : Algebra α)
    (seed offset : Int) (B : Nat) (u : State α) :
    run M ((body seed offset B).take 5) u = some (boundState seed offset B u) := by
  simp [body, run, step, evalExpr, evalOp_unfold, boundState]

theorem low_eval {α : Type} [Inhabited α] (M : Algebra α)
    (seed offset : Int) (B : Nat) (u : State α) :
    FP.Structural.evalOp M none (.intToUnsigned 32 (.intBitAnd 64 .nil
      (.ref .int [] "philox_offset") (.constInt 0xffffffff))) (boundState seed offset B u) =
        some (fun _ => (BitVec.ofInt 32 offset).toNat) := by
  apply TypedCounterOperands.eval_low_mask
  simp [evalOp_unfold, boundState]

theorem high_eval {α : Type} [Inhabited α] (M : Algebra α)
    (seed offset : Int) (B : Nat) (u : State α) :
    FP.Structural.evalOp M none (.intToUnsigned 32 (.intBitAnd 64 .nil
      (.intShiftRight .nil (.ref .int [] "philox_offset") (.constNat 32))
      (.constInt 0xffffffff))) (lowState seed offset B u) =
        some (fun _ => (counterHigh offset).toNat) := by
  apply TypedCounterOperands.eval_high_mask
  simp [evalOp_unfold, lowState, boundState]

theorem counters_exec {α : Type} [Inhabited α] (M : Algebra α)
    (seed offset : Int) (B : Nat) (u : State α) :
    run M ((body seed offset B).drop 5) (boundState seed offset B u) =
      some (counterState seed offset B u) := by
  simp only [body, List.drop_succ_cons, List.drop_zero, run, step, evalExpr]
  rw [low_eval]
  dsimp only [bind, Option.bind]
  have hh := high_eval M seed offset B u
  unfold lowState at hh
  rw [hh]
  rfl

theorem prefix_exec {α : Type} [Inhabited α] (M : Algebra α)
    (seed offset : Int) (B : Nat) (u : State α) :
    run M (body seed offset B) u = some (counterState seed offset B u) := by
  rw [← List.take_append_drop 5 (body seed offset B), run_append,
    bindings_exec, Option.bind_some, counters_exec]

theorem assign_step_of_eval {α : Type} [Inhabited α] (M : Algebra α)
    {dtype : TileDType} {shape : TileShape} (name : RegName) (e : Op dtype shape)
    (s : State α) (v : Values α dtype shape)
    (h : FP.Structural.evalOp M none e s = some v) :
    step M (.assign dtype shape name (.alg e)) s = some (s.setReg name dtype shape v) := by
  simp only [step, evalExpr, h, bind, Option.bind, pure]

theorem three_steps {α : Type} [Inhabited α] (M : Algebra α)
    {a b c : ComputeStmt} {s0 s1 s2 s3 : State α}
    (ha : step M a s0 = some s1) (hb : step M b s1 = some s2)
    (hc : step M c s2 = some s3) :
    run M [a, b, c] s0 = some s3 := by
  simp only [run, ha, hb, hc, bind, Option.bind, pure]

#axiomsClean source_prefix
#axiomsClean bindings_exec
#axiomsClean low_eval
#axiomsClean high_eval
#axiomsClean counters_exec
#axiomsClean prefix_exec
#axiomsClean assign_step_of_eval
#axiomsClean three_steps

end
end VeriTile.Bench.TritonBenchG.UniformSampling.TypedBindings
