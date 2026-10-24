import bench.repairs.tritonbench_g.rowwise_quantization_triton.Kernels
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Contract
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.Native
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.FP

/-! FP32 loads/reduction/division/multiplication, signed64 llrint followed by
low-eight-bit storage, and a separate direct FP16 maximum store. -/
namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.HistoricalFloating
open Direct
noncomputable section

/-- Exact source-order returned pair, without hardware numerical assumptions. -/
theorem rowwise_fp32 (allocate : Allocator) (native : NativeScalars)
    (x : Direct.Tensor) (laws : Direct.TensorLaws x) (h : InputContract x.metadata)
    (before : State) (resident : Resident x laws h before) (tree : Tree)
    (cover : tree.leaves.Perm (List.range (paddedWidth x.metadata)))
    (schedule : Schedule (Direct.launch x.metadata h tree cover))
    (q m : RowwiseQuantizationTriton.Tensor) (after : Heap)
    (success : checkedForward allocate (nativeOperations native) x laws h before resident tree cover schedule=.ok (some (q,m,after))) :
    ResultCorrect (nativeOperations native) (Direct.launch x.metadata h tree cover)
      (input x laws before) before q m after :=
  checked_correct allocate _ x laws h before resident tree cover schedule q m after success

/-- Deferred agreements are individual scalar primitives. No premise supplies
a row maximum, output row, complete kernel result or returned Tensor value. -/
theorem rowwise_equiv_if_local (allocate : Allocator) (native : NativeScalars)
    (reference : Operations Word) (localLaws : PrimitiveAgreement (nativeOperations native) reference)
    (x : Direct.Tensor) (laws : Direct.TensorLaws x) (h : InputContract x.metadata)
    (before : State) (resident : Resident x laws h before) (tree : Tree)
    (cover : tree.leaves.Perm (List.range (paddedWidth x.metadata)))
    (schedule : Schedule (Direct.launch x.metadata h tree cover))
    (q m : RowwiseQuantizationTriton.Tensor) (after : Heap)
    (success : checkedForward allocate (nativeOperations native) x laws h before resident tree cover schedule=.ok (some (q,m,after))) :
    ResultCorrect reference (Direct.launch x.metadata h tree cover) (input x laws before) before q m after := by
  have result := rowwise_fp32 allocate native x laws h before resident tree cover schedule q m after success
  rw [localLaws.operations] at result
  exact result

#axiomsClean rowwise_fp32
#axiomsClean rowwise_equiv_if_local
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.HistoricalFloating
