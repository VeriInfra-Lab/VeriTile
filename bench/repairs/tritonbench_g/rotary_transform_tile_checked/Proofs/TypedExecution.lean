import bench.repairs.tritonbench_g.rotary_transform_tile_checked.Proofs.TypedStores
import bench.repairs.tritonbench_g.rotary_transform_tile_checked.Proofs.TypedControl

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.TypedExecution
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 300000
set_option maxRecDepth 24000

/-- Composition uses slices of the same quotation. No shared source prefix is
reinterpreted by a handwritten program. -/
theorem payload_run {α : Type} [Inhabited α] (M : Algebra α) (s : Shape)
    (initial : State α) :
    run M (payload s) (indices M s initial).val = some (finish M s initial).val := by
  conv_lhs => rw [← List.take_append_drop (loadsCount s) (payload s)]
  rw [run_append]
  calc
    _ = (some (loads M s initial).val).bind (run M ((payload s).drop (loadsCount s))) :=
      congrArg (fun value : Option (State α) =>
        value.bind (run M ((payload s).drop (loadsCount s)))) (loads M s initial).property
    _ = some (finish M s initial).val := (finish M s initial).property

theorem continuation_run {α : Type} [Inhabited α] (M : Algebra α) (s : Shape)
    (initial : State α) :
    run M (continuation s) (setup M s Bool.true Bool.true initial).val =
      some (finish M s initial).val := by
  conv_lhs => rw [← List.take_append_drop 4 (continuation s)]
  rw [run_append]
  calc
    _ = (some (indices M s initial).val).bind (run M ((continuation s).drop 4)) :=
      congrArg (fun value : Option (State α) =>
        value.bind (run M ((continuation s).drop 4))) (indices M s initial).property
    _ = run M (payload s) (indices M s initial).val := continuation_tail M s _
    _ = some (finish M s initial).val := payload_run M s initial

def resultState {α : Type} [Inhabited α] (M : Algebra α) (s : Shape)
    (initial : State α) : State α :=
  if ((initial.pids 0*s.blockM : Nat) : Int) < ProgramModel.sequenceLength s initial
  then (finish M s initial).val else (setup M s Bool.true Bool.true initial).val

/-- The complete quotation successfully executes on every structural input.
The only branches are actual source branches; the witness is constructed from
their executions, without a source-success or completed-output premise. -/
theorem execution_result {α : Type} [Inhabited α] (M : Algebra α) (s : Shape)
    (cumulative64 offsets64 : Bool) (initial : State α) :
    FP.Structural.exec M (kernel s cumulative64 offsets64) initial =
      some (resultState M s initial) := by
  rw [kernel_carrier s cumulative64 offsets64]
  change run M (kernel s Bool.true Bool.true).surfaceBody initial = _
  rw [run_source_body]
  conv_lhs => rw [← List.take_append_drop (SourceBlocks.setupCount s.varlen) (sourceBody s)]
  rw [run_append]
  calc
    _ = (some (setup M s Bool.true Bool.true initial).val).bind
          (run M ((sourceBody s).drop (SourceBlocks.setupCount s.varlen))) :=
      congrArg (fun value : Option (State α) =>
        value.bind (run M ((sourceBody s).drop (SourceBlocks.setupCount s.varlen))))
        (setup M s Bool.true Bool.true initial).property
    _ = some (resultState M s initial) := by
      simp only [Option.bind_some,source_tail,run_if,guard_eval,Option.bind_some]
      by_cases active : ((initial.pids 0*s.blockM : Nat) : Int) < ProgramModel.sequenceLength s initial
      · simp only [active,decide_true,if_true,resultState,continuation_run]
      · simp only [active,decide_false,Bool.false_eq_true,if_false,resultState]

theorem executes {α : Type} [Inhabited α] (M : Algebra α) (s : Shape)
    (cumulative64 offsets64 : Bool) (initial : State α) :
    ∃ final, FP.Structural.exec M (kernel s cumulative64 offsets64) initial = some final :=
  ⟨resultState M s initial,execution_result M s cumulative64 offsets64 initial⟩

#axiomsClean payload_run
#axiomsClean continuation_run
#axiomsClean execution_result
#axiomsClean executes
end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.TypedExecution
