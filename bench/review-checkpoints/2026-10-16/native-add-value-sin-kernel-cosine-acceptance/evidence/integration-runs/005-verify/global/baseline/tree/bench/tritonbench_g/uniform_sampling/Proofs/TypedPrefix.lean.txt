import bench.tritonbench_g.uniform_sampling.Proofs.TypedIndex
import bench.tritonbench_g.uniform_sampling.Proofs.TypedTuple

/-! Complete typed execution through the actual Philox tuple statement.
All four output word registers are derived from the original signed64 seed
and offset, including inactive lanes and uint32 low-counter wrap.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.TypedPrefix
open VeriTile.Triton TypedSource FP.Structural
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 10000

def result {α : Type} (seed offset : Int) (B : Nat) (u : State α) : State α :=
  TypedTuple.outputState seed offset (TypedIndex.indices B u)
    (TypedTuple.capturedState seed offset (TypedIndex.indices B u)
      (TypedIndex.preparedState seed offset B u))

theorem tuple_exec {α : Type} [Inhabited α] (M : Algebra α)
    (seed offset : Int) (B : Nat) (u : State α)
    (hseed : Random.IntegerType.i64.inRange seed = Bool.true) :
    step M (TypedTuple.tupleBlock B) (TypedIndex.preparedState seed offset B u) =
      some (result seed offset B u) := by
  apply TypedTuple.tuple_exec M seed offset (TypedIndex.indices B u) _ hseed
  all_goals simp [TypedIndex.preparedState, TypedIndex.wrappedState, TypedIndex.indexState,
    TypedBindings.counterState, TypedBindings.lowState, TypedBindings.boundState]

theorem source_split (storage : Storage) (swapped : Bool) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : ℝ) (B : Nat)
    (hB : SourceDomain.LegalBlock B) :
    (source storage swapped y N seed offset lower upper B hB).surfaceBody.take 11 =
      (source storage swapped y N seed offset lower upper B hB).surfaceBody.take 10 ++
        [TypedTuple.tupleBlock B] := by
  cases storage <;> cases swapped <;> rfl

theorem prefix_exec {α : Type} [Inhabited α] (M : Algebra α)
    (storage : Storage) (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (hB : SourceDomain.LegalBlock B) (u : State α)
    (hseed : Random.IntegerType.i64.inRange seed = Bool.true) :
    run M ((source storage swapped y N seed offset lower upper B hB).surfaceBody.take 11) u =
      some (result seed offset B u) := by
  rw [source_split, run_append, TypedIndex.prefix_exec, Option.bind_some]
  simp only [run, tuple_exec M seed offset B u hseed, bind, Option.bind, pure]

theorem result_register {α : Type} (seed offset : Int) (B : Nat) (u : State α)
    (f : Random.Direct.Field) :
    (result seed offset B u).regs .nat [B] (TupleExecution.fieldName f) =
      some (fun i => (ScalarSpec.word seed offset (TypedIndex.indices B u i) f).toNat) :=
  TypedTuple.output_register seed offset (TypedIndex.indices B u) _ f

theorem block_register {α : Type} (seed offset : Int) (B : Nat) (u : State α) :
    (result seed offset B u).regs .nat [] "BLOCK_arg" = some (fun _ => B) := by
  simp [result, TypedTuple.outputState, TypedTuple.fieldState, TupleExecution.fieldName,
    TypedTuple.capturedState, TypedIndex.preparedState, TypedIndex.wrappedState,
    TypedIndex.indexState, TypedBindings.counterState, TypedBindings.lowState,
    TypedBindings.boundState]

theorem pids_frame {α : Type} (seed offset : Int) (B : Nat) (u : State α) :
    (result seed offset B u).pids = u.pids := by
  simp [result, TypedTuple.outputState, TypedTuple.fieldState, TypedTuple.capturedState,
    TypedIndex.preparedState, TypedIndex.wrappedState, TypedIndex.indexState,
    TypedBindings.counterState, TypedBindings.lowState, TypedBindings.boundState]

#axiomsClean tuple_exec
#axiomsClean source_split
#axiomsClean prefix_exec
#axiomsClean result_register
#axiomsClean block_register
#axiomsClean pids_frame

end
end VeriTile.Bench.TritonBenchG.UniformSampling.TypedPrefix
