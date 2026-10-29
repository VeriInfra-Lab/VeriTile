import bench.tritonbench_g.uniform_sampling.Proofs.TypedPrefix
import bench.tritonbench_g.uniform_sampling.Proofs.TypedAffine
import bench.tritonbench_g.uniform_sampling.Proofs.TypedOffsets

/-! The complete nineteen-statement typed prefix. Random words are symbolic
canonical Philox results; the floating algebra receives their actual fp32
uniform payloads, followed by the original fp32 affine expressions.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.TypedValues
open VeriTile.Triton TypedSource FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

def words {α : Type} (seed offset : Int) (B : Nat) (u : State α)
    (f : Random.Direct.Field) : TileIndex [B] → Random.Word :=
  fun i => ScalarSpec.word seed offset (TypedIndex.indices B u i) f

def valueState {α : Type} (M : Algebra α) (seed offset : Int) (lower upper : ℝ)
    (B : Nat) (u : State α) : State α :=
  TypedAffine.result M (words seed offset B u) lower upper (TypedPrefix.result seed offset B u)

def result {α : Type} (M : Algebra α) (seed offset : Int) (lower upper : ℝ)
    (B : Nat) (u : State α) : State α :=
  TypedOffsets.result (u.pids 0) B (valueState M seed offset lower upper B u)

theorem affines_exec {α : Type} [Inhabited α] (M : Algebra α)
    (seed offset : Int) (lower upper : ℝ) (B : Nat) (u : State α) :
    run M (TypedAffine.body B lower upper) (TypedPrefix.result seed offset B u) =
      some (valueState M seed offset lower upper B u) := by
  exact TypedAffine.affines_exec M (words seed offset B u) lower upper _
    (TypedPrefix.result_register seed offset B u)

theorem value_block_register {α : Type} (M : Algebra α) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (u : State α) :
    (valueState M seed offset lower upper B u).regs .nat [] "BLOCK_arg" = some (fun _ => B) := by
  simpa [valueState, TypedAffine.result] using TypedPrefix.block_register seed offset B u

theorem value_pids {α : Type} (M : Algebra α) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (u : State α) :
    (valueState M seed offset lower upper B u).pids = u.pids := by
  simpa [valueState, TypedAffine.result] using TypedPrefix.pids_frame seed offset B u

theorem offsets_exec {α : Type} [Inhabited α] (M : Algebra α)
    (seed offset : Int) (lower upper : ℝ) (B : Nat) (u : State α) :
    run M (TypedOffsets.body B) (valueState M seed offset lower upper B u) =
      some (result M seed offset lower upper B u) := by
  have he := TypedOffsets.offsets_exec M B (valueState M seed offset lower upper B u)
    (value_block_register M seed offset lower upper B u)
  rw [value_pids] at he
  exact he

theorem result_values {α : Type} (M : Algebra α) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (u : State α) (f : Random.Direct.Field) :
    (result M seed offset lower upper B u).regs .real [B] (TupleExecution.fieldName f) =
      some (TypedAffine.values M (words seed offset B u f) lower upper) := by
  have hv := TypedAffine.result_register M (words seed offset B u) lower upper
    (TypedPrefix.result seed offset B u) f
  cases f <;> simpa [result, TypedOffsets.result, valueState, TupleExecution.fieldName] using hv

theorem result_offsets {α : Type} (M : Algebra α) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (u : State α) :
    (result M seed offset lower upper B u).regs .nat [B] "off_0" =
        some (OffsetExecution.outputFn (u.pids 0) B 0) ∧
    (result M seed offset lower upper B u).regs .nat [B] "off_1" =
        some (OffsetExecution.outputFn (u.pids 0) B 1) ∧
    (result M seed offset lower upper B u).regs .nat [B] "off_2" =
        some (OffsetExecution.outputFn (u.pids 0) B 2) ∧
    (result M seed offset lower upper B u).regs .nat [B] "off_3" =
        some (OffsetExecution.outputFn (u.pids 0) B 3) := by
  simp [result, TypedOffsets.result]

theorem source_split (storage : Storage) (swapped : Bool) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : ℝ) (B : Nat)
    (hB : SourceDomain.LegalBlock B) :
    (source storage swapped y N seed offset lower upper B hB).surfaceBody.take 19 =
      ((source storage swapped y N seed offset lower upper B hB).surfaceBody.take 11 ++
        TypedAffine.body B lower upper) ++ TypedOffsets.body B := by
  cases storage <;> cases swapped <;> rfl

theorem prefix_exec {α : Type} [Inhabited α] (M : Algebra α)
    (storage : Storage) (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (hB : SourceDomain.LegalBlock B) (u : State α)
    (hseed : Random.IntegerType.i64.inRange seed = Bool.true) :
    run M ((source storage swapped y N seed offset lower upper B hB).surfaceBody.take 19) u =
      some (result M seed offset lower upper B u) := by
  rw [source_split, run_append, run_append, TypedPrefix.prefix_exec M storage swapped y N
    seed offset lower upper B hB u hseed, Option.bind_some, affines_exec,
    Option.bind_some, offsets_exec]

#axiomsClean affines_exec
#axiomsClean value_block_register
#axiomsClean value_pids
#axiomsClean offsets_exec
#axiomsClean result_values
#axiomsClean result_offsets
#axiomsClean source_split
#axiomsClean prefix_exec

end
end VeriTile.Bench.TritonBenchG.UniformSampling.TypedValues
