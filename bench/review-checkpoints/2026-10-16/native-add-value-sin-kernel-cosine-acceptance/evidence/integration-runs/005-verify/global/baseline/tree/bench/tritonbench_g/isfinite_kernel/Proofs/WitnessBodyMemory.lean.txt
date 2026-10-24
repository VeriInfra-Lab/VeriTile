import bench.tritonbench_g.isfinite_kernel.Proofs.WitnessBody

/-! Project one proven source iteration onto memory before composing states.
The input memory is a snapshot taken before the store fold. Only active lanes
are observed, so the unspecified padding oracle need not be fixed or erased. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
open VeriTile.Triton FP.Structural BitClassification
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def gridBodyMemory {α : Type} [Inhabited α] (classify : α → Bool) (j : Nat)
    (input : RegionName → Nat → Cell α) : RegionName → Nat → Cell α :=
  (TileShape.allIndices [8]).foldl (fun mem i =>
    if j * 8 + i.1.val < 14 then
      fun r o => if r = "output" ∧ o = (j * 8 + i.1.val) * 2 then
        .mk .bool (classify ((input "input" (j * 8 + i.1.val)).read .real))
      else mem r o
    else mem) input

theorem grid32_body_memory (rest : Algebra Float32Bits) (j : Nat)
    (s : State Float32Bits) (h : GridReady s) :
    (grid32_body_run rest j s h).val.mem = gridBodyMemory finite32 j s.mem := by
  simp [grid32_body_run, fold_writes_mem, gridBodyMemory,
    TileShape.indexToList, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]
  congr 1
  funext mem i
  by_cases hm : j * 8 + i.1.val < 14 <;> simp only [hm, ite_true, ite_false]

theorem grid64_body_memory (rest : Algebra Float64Bits) (j : Nat)
    (s : State Float64Bits) (h : GridReady s) :
    (grid64_body_run rest j s h).val.mem = gridBodyMemory finite64 j s.mem := by
  simp [grid64_body_run, fold_writes_mem, gridBodyMemory,
    TileShape.indexToList, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]
  congr 1
  funext mem i
  by_cases hm : j * 8 + i.1.val < 14 <;> simp only [hm, ite_true, ite_false]

#axiomsClean grid32_body_memory
#axiomsClean grid64_body_memory
end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
