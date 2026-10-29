import bench.tritonbench_g.isfinite_kernel.Proofs.Body

/-! A single full source body has exactly this masked, strided memory effect.
The classified input is read from the incoming memory snapshot. This remains
valid even for aliased regions or colliding output addresses: the actual lane
order is retained. Out-of-bounds load padding never affects an active store. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.Parametric
open VeriTile.Triton FP.Structural TypedBits BitClassification
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def tileMemory {α : Type} [Inhabited α] (classify : α → Bool)
    (p : Parameters) (base : Nat) (input : RegionName → Nat → Cell α) :
    RegionName → Nat → Cell α :=
  (TileShape.allIndices [p.block]).foldl (fun mem i =>
    if base + i.1.val < p.size then
      fun r o => if r = p.output ∧ o = (base + i.1.val) * p.outputStride then
        .mk .bool (classify ((input p.input
          ((base + i.1.val) * p.inputStride)).read .real))
      else mem r o
    else mem) input

theorem one32_memory (rest : Algebra Float32Bits) (p : Parameters) (pid : Nat)
    (s : State Float32Bits) (h : OneReady pid s) :
    (one32_run rest p pid s h).val.mem =
      tileMemory finite32 p (pid * p.block) s.mem := by
  simp [one32_run, fold_writes_mem, tileMemory,
    TileShape.indexToList, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]
  congr 1
  funext mem i
  by_cases hm : pid * p.block + i.1.val < p.size <;>
    simp only [hm, ite_true, ite_false]

theorem one64_memory (rest : Algebra Float64Bits) (p : Parameters) (pid : Nat)
    (s : State Float64Bits) (h : OneReady pid s) :
    (one64_run rest p pid s h).val.mem =
      tileMemory finite64 p (pid * p.block) s.mem := by
  simp [one64_run, fold_writes_mem, tileMemory,
    TileShape.indexToList, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]
  congr 1
  funext mem i
  by_cases hm : pid * p.block + i.1.val < p.size <;>
    simp only [hm, ite_true, ite_false]

theorem loop32_memory (rest : Algebra Float32Bits) (p : Parameters)
    (pid numCtas j : Nat) (s : State Float32Bits) (h : LoopReady pid numCtas s) :
    (loop32_run rest p pid numCtas j s h).val.mem =
      tileMemory finite32 p ((pid + j * numCtas) * p.block) s.mem := by
  simp [loop32_run, fold_writes_mem, tileMemory,
    TileShape.indexToList, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]
  congr 1
  funext mem i
  by_cases hm : (pid + j * numCtas) * p.block + i.1.val < p.size <;>
    simp only [hm, ite_true, ite_false]

theorem loop64_memory (rest : Algebra Float64Bits) (p : Parameters)
    (pid numCtas j : Nat) (s : State Float64Bits) (h : LoopReady pid numCtas s) :
    (loop64_run rest p pid numCtas j s h).val.mem =
      tileMemory finite64 p ((pid + j * numCtas) * p.block) s.mem := by
  simp [loop64_run, fold_writes_mem, tileMemory,
    TileShape.indexToList, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]
  congr 1
  funext mem i
  by_cases hm : (pid + j * numCtas) * p.block + i.1.val < p.size <;>
    simp only [hm, ite_true, ite_false]

#axiomsClean one32_memory
#axiomsClean one64_memory
#axiomsClean loop32_memory
#axiomsClean loop64_memory
end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.Parametric
