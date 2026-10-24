import bench.tritonbench_g.isfinite_kernel.Proofs.WitnessOne
import bench.tritonbench_g.isfinite_kernel.Proofs.WitnessValues

/-! Complete source execution over all fourteen special-value examples.
The one-tile arm uses sixteen lanes; the grid arm executes two eight-lane
iterations. Both have a partially active final tile. Output stride two exposes
untouched holes, and the initial omitted-padding oracle returns signaling NaN.
All statements, including both pointer casts and the source helper, come from
IsfiniteWholeSource. No native acceptance is inferred from these witnesses. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource BitClassification
noncomputable section
set_option maxHeartbeats 1600000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def full32 (oneTile : Bool) : ComputeKernel :=
  source32 "input" "output" 1 0 2 0 14 14 2
    (if oneTile then 16 else 8) oneTile rfl rfl

def full64 (oneTile : Bool) : ComputeKernel :=
  source64 "input" "output" 1 0 2 0 14 14 2
    (if oneTile then 16 else 8) oneTile rfl rfl

def expectedObservation : List (Option Bool) × List (Option Nat) :=
  (expected.map some, List.replicate 17 (some 1729))

def observation {α : Type} (s : State α) :
    List (Option Bool) × List (Option Nat) :=
  (outputValues s, sentinelValues s)

theorem whole32_one_tile_samples (rest : Algebra Float32Bits) :
    (FP.Structural.exec (engine (backend32 rest)) (full32 Bool.true) initial32).map observation =
      some expectedObservation := by
  simp [full32, source32, FP.Structural.exec, ComputeKernel.surfaceBody, FP.Structural.run, FP.Structural.step,
    FP.Structural.evalExpr, FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold, FP.Structural.evalOffsets, FP.Structural.store,
    FP.Structural.numeric, FP.Structural.bop, engine, Algebra.withDefaultPrecision, resolvePrecision,
    backend32, initial32, State.setReg, State.write, Cell.read,
    observation, outputValues, sentinelValues, boolCell, natCell,
    ComputeDType.eraseDType, FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast,
    TileShape.allIndices, TileShape.indexToList,
    finite32, exponent32, bits32, sample32, expectedObservation, expected,
    Option.bind, List.finRange_succ, List.range_succ] <;> decide

theorem whole64_one_tile_samples (rest : Algebra Float64Bits) :
    (FP.Structural.exec (engine (backend64 rest)) (full64 Bool.true) initial64).map observation =
      some expectedObservation := by
  simp [full64, source64, FP.Structural.exec, ComputeKernel.surfaceBody, FP.Structural.run, FP.Structural.step,
    FP.Structural.evalExpr, FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold, FP.Structural.evalOffsets, FP.Structural.store,
    FP.Structural.numeric, FP.Structural.bop, engine, Algebra.withDefaultPrecision, resolvePrecision,
    backend64, initial64, State.setReg, State.write, Cell.read,
    observation, outputValues, sentinelValues, boolCell, natCell,
    ComputeDType.eraseDType, FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast,
    TileShape.allIndices, TileShape.indexToList,
    finite64, exponent64, bits64, sample64, expectedObservation, expected,
    Option.bind, List.finRange_succ, List.range_succ] <;> decide

theorem whole32_grid_samples (rest : Algebra Float32Bits) :
    (FP.Structural.exec (engine (backend32 rest)) (full32 Bool.false) initial32).map observation =
      some expectedObservation := by
  change (FP.Structural.exec (engine (backend32 rest)) gridKernel32 initial32).map observation = _
  rw [(grid32_run rest).property]
  simp only [Option.map_some, Option.some.injEq]
  change memoryObservation (grid32_run rest).val.mem = expectedObservation
  rw [(grid32_memory rest).property]
  exact grid32_memory_values rest

theorem whole64_grid_samples (rest : Algebra Float64Bits) :
    (FP.Structural.exec (engine (backend64 rest)) (full64 Bool.false) initial64).map observation =
      some expectedObservation := by
  change (FP.Structural.exec (engine (backend64 rest)) gridKernel64 initial64).map observation = _
  rw [(grid64_run rest).property]
  simp only [Option.map_some, Option.some.injEq]
  change memoryObservation (grid64_run rest).val.mem = expectedObservation
  rw [(grid64_memory rest).property]
  exact grid64_memory_values rest

/-- An observed result entails successful complete execution. The result is
not supplied as a prepared state: it is extracted from the actual FP.Structural.run. -/
theorem exists_state_of_observation {α : Type} (result : Option (State α))
    (h : result.map observation = some expectedObservation) :
    ∃ s, result = some s ∧ observation s = expectedObservation := by
  cases result with
  | none => simp at h
  | some s =>
    refine ⟨s, rfl, ?_⟩
    exact Option.some.inj h

theorem successful32_both_branches (rest : Algebra Float32Bits) (oneTile : Bool) :
    ∃ s, FP.Structural.exec (engine (backend32 rest)) (full32 oneTile) initial32 = some s ∧
      observation s = expectedObservation := by
  cases oneTile
  · exact exists_state_of_observation _ (whole32_grid_samples rest)
  · exact exists_state_of_observation _ (whole32_one_tile_samples rest)

theorem successful64_both_branches (rest : Algebra Float64Bits) (oneTile : Bool) :
    ∃ s, FP.Structural.exec (engine (backend64 rest)) (full64 oneTile) initial64 = some s ∧
      observation s = expectedObservation := by
  cases oneTile
  · exact exists_state_of_observation _ (whole64_grid_samples rest)
  · exact exists_state_of_observation _ (whole64_one_tile_samples rest)

#axiomsClean whole32_one_tile_samples
#axiomsClean whole64_one_tile_samples
#axiomsClean whole32_grid_samples
#axiomsClean whole64_grid_samples
#axiomsClean exists_state_of_observation
#axiomsClean successful32_both_branches
#axiomsClean successful64_both_branches
end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
