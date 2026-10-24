import bench.repairs.tritonbench_g.seeded_dropout.Proofs.Contract
import bench.tritonbench_g.seeded_dropout.Proofs.SourceDomain
import VeriTile.Torch.FP32
import VeriTile.Triton.Memory.Bounds

namespace VeriTile.Bench.Repairs.SeededDropout.KernelMemory
open VeriTile.Triton VeriTile.Torch
open TritonBenchG.SeededDropout

/-- Ordinary capacities of the two typed memory regions. Only active accesses
need storage; omitted tail padding is unchanged. -/
structure Bounds (X O : RegionName) (n : Nat) (capacity : RegionBounds) : Prop where
  input : n ≤ capacity X
  output : n ≤ capacity O

theorem active_indices (X O : RegionName) (n pid : Nat) (capacity : RegionBounds)
    (bounds : Bounds X O n capacity) (hn : n < 2^31) (hp : pid < gridCount n)
    (lane : Fin 1024) :
    (SourceDomain.offsetBits pid 1024 lane.val).toNat = pid*1024+lane.val ∧
    ((SourceDomain.offsetBits pid 1024 lane.val).slt (SourceDomain.i32 n) = Bool.true ↔
      pid*1024+lane.val < n) ∧
    (pid*1024+lane.val < n →
      pid*1024+lane.val < capacity X ∧ pid*1024+lane.val < capacity O) := by
  refine ⟨(SourceDomain.source_operations_exact SourceDomain.wrapper_block_legal hn hp lane.isLt).2.2.2.2.2,
    SourceDomain.source_mask_exact SourceDomain.wrapper_block_legal hn hp lane.isLt, ?_⟩
  intro active
  exact ⟨lt_of_lt_of_le active bounds.input,lt_of_lt_of_le active bounds.output⟩

/-- Checked contiguity, negative-view and pointer alignment guards connect to
shared Tensor storage laws, without allocator or launch behavior. -/
theorem checked_input_memory (t : FP32.Tensor) (laws : FP32.TensorLaws t)
    (p : ℝ) (seed : Int) (admitted : InputContract t.metadata p seed)
    (read : Nat → FP32.Word) :
    let view := FP32.dataView t laws admitted.tensor.contiguous.1
    view.base % 4 = 0 ∧
    (0 < t.metadata.numel → BitVec.ofNat 64 (view.address 0) = t.dataPtr) ∧
    (∀ i, i < t.metadata.numel → read (view.address i) = FP32.logicalValue read t laws i) := by
  refine ⟨FP32.dataView_aligned t laws admitted.tensor.contiguous.1
    (fun hn => (admitted.tensor.alignment hn).1),?_,?_⟩
  · exact FP32.actual_pointer t laws admitted.tensor.contiguous.1
  · exact FP32.direct_read_eq_logical read t laws admitted.tensor.contiguous.1 admitted.tensor.noNeg.1

#axiomsClean active_indices
#axiomsClean checked_input_memory
end VeriTile.Bench.Repairs.SeededDropout.KernelMemory
