import bench.repairs.tritonbench_g.sin_kernel.Proofs.FP

namespace VeriTile.Bench.Repairs.SinKernel.FPEquiv
open VeriTile.Triton
open Binding (Word)
noncomputable section

/-- End-to-end source result on actual Tensor storage. `h` is exactly the Python
input contract. The other premises are ordinary Tensor/allocator/load/store and
successful launch contracts, not additional input restrictions or output laws.
The unchanged finite source executes against the real data pointers, and the
returned Tensor's logical values use entry-state logical input values. -/
specification sin_wrapper_tensor_correct {Memory : Type}
    (ops : VeriTile.Bench.TritonBenchG.AddExample.Finite.MemoryPrimitives Memory Word)
    (_storage : FPProof.FP32Storage ops)
    (nativeSin : Word → Word) (x : Binding.Tensor)
    (xTensor : Binding.TensorLaws x)
    (h : InputContract x.metadata) (before : Memory)
    (allocation : Binding.EmptyLikeResult ops x h before)
    (runtime : Binding.Runtime ops x xTensor h before allocation)
    (memory : Native.LocalMemory ops (Binding.layout ops x xTensor h before allocation))
    (schedule : Index.Schedule x.metadata.numel) :
    ∃ final,
      Binding.runWrapper ops nativeSin x xTensor h before allocation schedule.programs =
        .ok (some (allocation.output, final)) ∧
      (∀ i, i < x.metadata.numel → Binding.logicalValue ops final allocation.output allocation.laws i =
        nativeSin (Binding.logicalValue ops before x xTensor i)) ∧
      VeriTile.Bench.TritonBenchG.AddExample.Finite.Preserved ops
        (Native.view (Binding.dataView allocation.output allocation.laws allocation.contiguous) 0) final before ∧
      (0 < x.metadata.numel →
        (Native.view (Binding.dataView x xTensor h.contiguous) 0).pointer = x.dataPtr ∧
          (Native.view (Binding.dataView allocation.output allocation.laws allocation.contiguous) 0).pointer =
          allocation.output.dataPtr) ∧
      ops.launchDevice = x.metadata.device.index ∧
      allocation.output.metadata.shape = x.metadata.shape ∧
      allocation.output.metadata.device = x.metadata.device ∧
      allocation.output.metadata.dtype = .floating .fp32 := by
  obtain ⟨final, he, hv, hf⟩ := Binding.wrapper_correct ops nativeSin x xTensor
    h before allocation memory schedule
  refine ⟨final, he, hv, hf, ?_, runtime.currentDevice, ?_, ?_, ?_⟩
  · intro nonempty
    exact ⟨Binding.actual_pointer x xTensor h.contiguous nonempty,
      Binding.actual_pointer allocation.output allocation.laws allocation.contiguous
        (by simpa [allocation.metadata, VeriTile.Torch.TensorMetadata.numel] using nonempty)⟩
  · simp [allocation.metadata]
  · simp [allocation.metadata]
  · simpa [allocation.metadata] using h.dtype

/-- The same actual source and returned Tensor match the independent FP32
reference if its single sine primitive conforms. The domain remains all FP32
words; there is no hidden finite-input guard. -/
specification sin_wrapper_reference_if_local {Memory : Type}
    (ops : TritonBenchG.AddExample.Finite.MemoryPrimitives Memory Word)
    (_storage : FPProof.FP32Storage ops)
    (nativeSin : Word → Word) (target : FP.Structural.Algebra Word)
    (numerical : FPProof.LocalSine nativeSin target)
    (x : Binding.Tensor) (xTensor : Binding.TensorLaws x)
    (h : InputContract x.metadata) (before : Memory)
    (allocation : Binding.EmptyLikeResult ops x h before)
    (memory : Native.LocalMemory ops (Binding.layout ops x xTensor h before allocation))
    (schedule : Index.Schedule x.metadata.numel) :
    ∃ final,
      Binding.runWrapper ops nativeSin x xTensor h before allocation schedule.programs =
        .ok (some (allocation.output, final)) ∧
      (∀ i, i < x.metadata.numel →
        Binding.logicalValue ops final allocation.output allocation.laws i =
          FPProof.reference target (Binding.logicalValue ops before x xTensor i)) ∧
      TritonBenchG.AddExample.Finite.Preserved ops
        (Native.view (Binding.dataView allocation.output allocation.laws allocation.contiguous) 0)
        final before := by
  obtain ⟨final, he, hv, hf⟩ := Binding.wrapper_correct ops nativeSin x xTensor
    h before allocation memory schedule
  exact ⟨final, he, fun i hi => (hv i hi).trans (numerical _), hf⟩

#axiomsClean sin_wrapper_tensor_correct
#axiomsClean sin_wrapper_reference_if_local
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SinKernel.FPEquiv
