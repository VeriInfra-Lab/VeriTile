import bench.tritonbench_g.isfinite_kernel.Proofs.WitnessRun
import bench.tritonbench_g.isfinite_kernel.Proofs.WitnessBodyMemory

/-! Memory-only projections of the actual successful grid runs. Their final
store folds are converted to folds over memory functions before finite sample
enumeration; register histories are absent from the resulting calculation. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
open VeriTile.Triton FP.Structural BitClassification
noncomputable section
set_option maxHeartbeats 1200000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def grid32_memory (rest : Algebra Float32Bits) :
    {mem : RegionName → Nat → Cell Float32Bits //
      (grid32_run rest).val.mem = mem} := by
  refine ⟨gridBodyMemory finite32 1 (gridBodyMemory finite32 0 initial32.mem), ?_⟩
  change (grid32_second rest).val.mem = _
  unfold grid32_second
  rw [grid32_body_memory]
  unfold grid32_first
  rw [grid32_body_memory]
  rfl

def grid64_memory (rest : Algebra Float64Bits) :
    {mem : RegionName → Nat → Cell Float64Bits //
      (grid64_run rest).val.mem = mem} := by
  refine ⟨gridBodyMemory finite64 1 (gridBodyMemory finite64 0 initial64.mem), ?_⟩
  change (grid64_second rest).val.mem = _
  unfold grid64_second
  rw [grid64_body_memory]
  unfold grid64_first
  rw [grid64_body_memory]
  rfl

def memoryObservation {α : Type} (mem : RegionName → Nat → Cell α) :
    List (Option Bool) × List (Option Nat) :=
  ((List.range 14).map (fun i => boolCell (mem "output" (2 * i))),
   (List.range 14).map (fun i => natCell (mem "output" (2 * i + 1))) ++
   [natCell (mem "output" 28), natCell (mem "output" 30),
    natCell (mem "unrelated" 0)])

theorem grid32_memory_values (rest : Algebra Float32Bits) :
    memoryObservation (grid32_memory rest).val =
      (expected.map some, List.replicate 17 (some 1729)) := by
  simp [memoryObservation, grid32_memory, gridBodyMemory,
    initial32, boolCell, natCell, Cell.read,
    TileShape.allIndices, TileShape.indexToList, List.finRange_succ, List.range_succ,
    finite32, exponent32, bits32, sample32, expected] <;> decide

theorem grid64_memory_values (rest : Algebra Float64Bits) :
    memoryObservation (grid64_memory rest).val =
      (expected.map some, List.replicate 17 (some 1729)) := by
  simp [memoryObservation, grid64_memory, gridBodyMemory,
    initial64, boolCell, natCell, Cell.read,
    TileShape.allIndices, TileShape.indexToList, List.finRange_succ, List.range_succ,
    finite64, exponent64, bits64, sample64, expected] <;> decide

#axiomsClean grid32_memory
#axiomsClean grid64_memory
#axiomsClean grid32_memory_values
#axiomsClean grid64_memory_values
end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
