import bench.repairs.tritonbench_g.sin_computation.Proofs.FP

namespace VeriTile.Bench.Repairs.SinComputation.Contiguous.FPEquiv
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AddExample.Finite
open Binding (Word)
noncomputable section

/-- Exact supplied-object mutation and Python None, every active logical value,
the full exterior byte frame (including output tail), and actual pointers.
The same contract admits empty, offset, larger output, and exact in-place use. -/
specification sin_tensor_correct {M : Type} (ops : MemoryPrimitives M Word)
    (_storage : FPProof.FP32Storage ops) (native : FPProof.Numerics)
    (x out : Binding.Tensor) (lx : Binding.TensorLaws x) (lo : Binding.TensorLaws out)
    (h : InputContract x.metadata out.metadata) (runtime : Binding.Runtime ops x)
    (memory : Native.LocalMemory ops (Binding.layout x out lx lo h))
    (schedule : Index.Schedule x.metadata.numel) (before : M) :
    ∃ after,
      Binding.runWrapper ops (FPProof.scalar native) x out lx lo h schedule.programs before =
        .ok (some (out,(),after)) ∧
      (∀ i,i<x.metadata.numel → Binding.logicalValue ops after out lo i =
        FPProof.scalar native (Binding.logicalValue ops before x lx i)) ∧
      Preserved ops (Native.view (Binding.window out lo h.contiguous.2 x.metadata.numel h.capacity) 0) after before ∧
      (0<x.metadata.numel →
        (Native.view (Binding.window x lx h.contiguous.1 x.metadata.numel (Nat.le_refl _)) 0).pointer = x.dataPtr ∧
        (Native.view (Binding.window out lo h.contiguous.2 x.metadata.numel h.capacity) 0).pointer = out.dataPtr) ∧
      ops.launchDevice = x.metadata.device.index := by
  obtain ⟨after,he,hv,hf⟩ := Binding.wrapper_correct ops (FPProof.scalar native) x out lx lo h memory schedule before
  exact ⟨after,he,hv,hf,fun positive =>
    ⟨Binding.actual_pointer x lx h.contiguous.1 _ (Nat.le_refl _) positive,
      Binding.actual_pointer out lo h.contiguous.2 _ h.capacity positive⟩,runtime.device⟩

/-- The only conditional refinement is the retained local all-word sine port. -/
specification sin_reference_if_local {M : Type} (ops : MemoryPrimitives M Word)
    (_storage : FPProof.FP32Storage ops) (native : FPProof.Numerics) (target : FPProof.Target)
    (numerical : FPProof.SineObligation native target)
    (x out : Binding.Tensor) (lx : Binding.TensorLaws x) (lo : Binding.TensorLaws out)
    (h : InputContract x.metadata out.metadata)
    (memory : Native.LocalMemory ops (Binding.layout x out lx lo h))
    (schedule : Index.Schedule x.metadata.numel) (before : M) :
    ∃ after,
      Binding.runWrapper ops (FPProof.scalar native) x out lx lo h schedule.programs before =
        .ok (some (out,(),after)) ∧
      (∀ i,i<x.metadata.numel → Binding.logicalValue ops after out lo i =
        FPProof.reference target (Binding.logicalValue ops before x lx i)) ∧
      Preserved ops (Native.view (Binding.window out lo h.contiguous.2 x.metadata.numel h.capacity) 0) after before := by
  obtain ⟨after,he,hv,hf⟩ := Binding.wrapper_correct ops (FPProof.scalar native) x out lx lo h memory schedule before
  exact ⟨after,he,fun i hi => (hv i hi).trans (FPProof.scalar_reference native target numerical _),hf⟩

#axiomsClean sin_tensor_correct
#axiomsClean sin_reference_if_local
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SinComputation.Contiguous.FPEquiv
