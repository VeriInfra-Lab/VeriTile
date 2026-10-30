import bench.repairs.tritonbench_g.quantize_global.Proofs.Direct
import VeriTile.Torch.FP32
/-! Ordinary typed-region/storage binding. Input cells are bound to the actual
entry heap and the original Tensor view; this law says nothing about output. -/
namespace QuantizeGlobalRepair.Direct
open VeriTile.Triton FP.Structural
noncomputable section
theorem source_output_pointer (x : TensorMetadata) (X : RegionName) (alloc : Allocation x X)
    (i : Nat) (hi : i<x.numel) :
    (alloc.output.metadata.dataPtr+BitVec.ofNat 64 i).toNat=alloc.output.metadata.dataPtr.toNat+i := by
  have exactPtr := pointer_exact alloc.output.metadata.dataPtr.toNat 1 x.numel i (by decide) (by simpa using alloc.outputSpan) hi
  simpa using exactPtr

theorem source_maximum_pointer (x : TensorMetadata) (X : RegionName) (alloc : Allocation x X) :
    alloc.maximum.metadata.dataPtr.toNat+4≤2^64 ∧ alloc.maximum.metadata.dataPtr.toNat%4=0 :=
  ⟨alloc.maximumSpan,alloc.maximumAlignment⟩

#axiomsClean original_logical_input
#axiomsClean source_input_pointer
#axiomsClean source_output_pointer
#axiomsClean source_maximum_pointer
#guard_msgs (drop info) in
#auditModuleAxioms
end
end QuantizeGlobalRepair.Direct
