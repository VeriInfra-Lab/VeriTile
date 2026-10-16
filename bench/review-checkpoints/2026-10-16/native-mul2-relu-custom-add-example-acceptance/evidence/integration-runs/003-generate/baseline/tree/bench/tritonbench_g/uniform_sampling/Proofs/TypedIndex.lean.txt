import bench.tritonbench_g.uniform_sampling.Proofs.TypedBindings
import bench.tritonbench_g.uniform_sampling.Proofs.Algorithm

/-! Typed evaluation of every counter lane, including genuine uint32 wrap.
The block size remains symbolic, so no Philox output or tile is precomputed.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.TypedIndex
open VeriTile.Triton ScalarSpec TypedSource FP.Structural TypedBindings
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 10000

def body (swapped : Bool) (B : Nat) : List ComputeStmt := [
  .assign .nat [B] "i4" (.alg (Algorithm.indexOp swapped B)),
  .assign .nat [B] "c0" (.alg (.intToUnsigned 32 (.add .int .scalarL
    (.castNatToInt (.ref .nat [] "c0")) (.castNatToInt (.ref .nat [B] "i4"))))),
  .assign .nat [B] "_O" (.alg (.intToUnsigned 32 (.mul .int .scalarR
    (.castNatToInt (.ref .nat [B] "c0")) (.constInt 0))))]

theorem source_slice (storage : Storage) (swapped : Bool) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : ℝ) (B : Nat)
    (hB : SourceDomain.LegalBlock B) :
    ((source storage swapped y N seed offset lower upper B hB).surfaceBody.drop 7).take 3 =
      body swapped B := by
  cases storage <;> cases swapped <;> rfl

def indices {α : Type} (B : Nat) (u : State α) : TileIndex [B] → Nat :=
  fun i => SourceDomain.counterIndex (u.pids 0) B i.1.val

def indexState {α : Type} (seed offset : Int) (B : Nat) (u : State α) : State α :=
  (counterState seed offset B u).setReg "i4" .nat [B] (indices B u)

def wrappedState {α : Type} (seed offset : Int) (B : Nat) (u : State α) : State α :=
  (indexState seed offset B u).setReg "c0" .nat [B]
    (fun i => (counterLow offset (indices B u i)).toNat)

def preparedState {α : Type} (seed offset : Int) (B : Nat) (u : State α) : State α :=
  (wrappedState seed offset B u).setReg "_O" .nat [B] (fun _ => 0)

theorem index_eval {α : Type} [Inhabited α] (M : Algebra α)
    (swapped : Bool) (seed offset : Int) (B : Nat) (u : State α) :
    FP.Structural.evalOp M none (Algorithm.indexOp swapped B) (counterState seed offset B u) =
      some (indices B u) := by
  cases swapped <;>
    simp [Algorithm.indexOp, evalOp_unfold, counterState, lowState, boundState,
      bop, numeric, Nat.mul_comm]
  all_goals funext i
  all_goals simp [bop, indices, SourceDomain.counterIndex, Nat.mul_comm]

theorem wrapped_eval {α : Type} [Inhabited α] (M : Algebra α)
    (seed offset : Int) (B : Nat) (u : State α) :
    FP.Structural.evalOp M none (.intToUnsigned 32 (.add .int .scalarL
      (.castNatToInt (.ref .nat [] "c0")) (.castNatToInt (.ref .nat [B] "i4"))))
      (indexState seed offset B u) =
        some (fun i => (counterLow offset (indices B u i)).toNat) := by
  have hl : (indexState seed offset B u).regs .nat [] "c0" =
      some (fun _ => (BitVec.ofInt 32 offset).toNat) := by
    simp [indexState, counterState, lowState]
  have hi : (indexState seed offset B u).regs .nat [B] "i4" = some (indices B u) := by
    simp [indexState]
  rw [FP.Structural.eval_intToUnsigned]
  simp only [evalOp_unfold, hl, hi, bind, Option.bind, pure, bop, numeric,
    Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL]
  exact congrArg some (funext fun i => CounterOperands.wrapped_low_add offset (indices B u i))

theorem zeros_eval {α : Type} [Inhabited α] (M : Algebra α)
    (seed offset : Int) (B : Nat) (u : State α) :
    FP.Structural.evalOp M none (.intToUnsigned 32 (.mul .int .scalarR
      (.castNatToInt (.ref .nat [B] "c0")) (.constInt 0)))
      (wrappedState seed offset B u) = some (fun _ => 0) := by
  rw [FP.Structural.eval_intToUnsigned]
  simp [evalOp_unfold, wrappedState, bop, numeric]

theorem preparation_exec {α : Type} [Inhabited α] (M : Algebra α)
    (swapped : Bool) (seed offset : Int) (B : Nat) (u : State α) :
    run M (body swapped B) (counterState seed offset B u) =
      some (preparedState seed offset B u) := by
  exact three_steps M
    (assign_step_of_eval M "i4" _ _ _ (index_eval M swapped seed offset B u))
    (assign_step_of_eval M "c0" _ _ _ (wrapped_eval M seed offset B u))
    (assign_step_of_eval M "_O" _ _ _ (zeros_eval M seed offset B u))

theorem source_split (storage : Storage) (swapped : Bool) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : ℝ) (B : Nat)
    (hB : SourceDomain.LegalBlock B) :
    (source storage swapped y N seed offset lower upper B hB).surfaceBody.take 10 =
      TypedBindings.body seed offset B ++ body swapped B := by
  cases storage <;> cases swapped <;> rfl

theorem prefix_exec {α : Type} [Inhabited α] (M : Algebra α)
    (storage : Storage) (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (hB : SourceDomain.LegalBlock B) (u : State α) :
    run M ((source storage swapped y N seed offset lower upper B hB).surfaceBody.take 10) u =
      some (preparedState seed offset B u) := by
  rw [source_split, run_append, TypedBindings.prefix_exec, Option.bind_some, preparation_exec]

#axiomsClean source_slice
#axiomsClean index_eval
#axiomsClean wrapped_eval
#axiomsClean zeros_eval
#axiomsClean preparation_exec
#axiomsClean source_split
#axiomsClean prefix_exec

end
end VeriTile.Bench.TritonBenchG.UniformSampling.TypedIndex
