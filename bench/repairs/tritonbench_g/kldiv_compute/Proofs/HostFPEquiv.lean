import bench.repairs.tritonbench_g.kldiv_compute.Proofs.ConditionalFPEquiv
import bench.repairs.tritonbench_g.kldiv_compute.Proofs.Checked.FP

namespace VeriTile.Bench.Repairs.KldivCompute.Contiguous.FPEquiv
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AddExample.Finite
open FPProof Binding
noncomputable section

/-- Exact returned first allocation, whole logical FP32 result, all exterior
caller bytes and actual pointer binding under the checked admission predicate.
Allocator/memory/runtime premises are local API facts, not KL output laws. -/
theorem checked_native {M : Type} (ops : MemoryPrimitives M FPProof.Word) (native : Numerics)
    (x y : Tensor) (lx : TensorLaws x) (ly : TensorLaws y)
    (h : InputContract x.metadata y.metadata) (before : M)
    (allocation : EmptyLikeResult ops x y h before)
    (runtime : Runtime ops x y lx ly h before allocation)
    (memory : Native.LocalMemory ops (layout ops x y lx ly h before allocation))
    (schedule : Index.Schedule x.metadata.numel) :
    ∃ final,runWrapper ops (formula native) x y lx ly h before allocation schedule.programs=
        .ok (some (allocation.output,final)) ∧
      (∀ i,i<x.metadata.numel → logicalValue ops final allocation.output allocation.laws i=
        formula native (logicalValue ops before x lx i) (logicalValue ops before y ly i)) ∧
      Preserved ops (Native.view (dataView allocation.output allocation.laws allocation.contiguous) 0) final before ∧
      (0<x.metadata.numel →
        (Native.view (dataView x lx h.contiguous.1) 0).pointer=x.dataPtr ∧
        (Native.view (dataView y ly h.contiguous.2) 0).pointer=y.dataPtr ∧
        (Native.view (dataView allocation.output allocation.laws allocation.contiguous) 0).pointer=allocation.output.dataPtr) ∧
      ops.launchDevice=x.metadata.device.index ∧ allocation.output.metadata.shape=x.metadata.shape ∧
      allocation.output.metadata.device=x.metadata.device ∧ allocation.output.metadata.dtype=.floating .fp32 ∧
      allocation.output.object≠x.object ∧ allocation.output.object≠y.object := by
  obtain ⟨final,execution,values,frame⟩ := wrapper_correct ops (formula native) x y lx ly h before allocation memory schedule
  refine ⟨final,execution,values,frame,?_,runtime.currentDevice,?_,?_,?_,allocation.freshObject⟩
  · intro nonempty
    exact ⟨actual_pointer x lx h.contiguous.1 nonempty,
      actual_pointer y ly h.contiguous.2 (by have same := same_numel x.metadata y.metadata h; omega),
      actual_pointer allocation.output allocation.laws allocation.contiguous
        (by simpa [allocation.metadata,VeriTile.Torch.TensorMetadata.numel] using nonempty)⟩
  · simp [allocation.metadata]
  · simp [allocation.metadata]
  · simpa [allocation.metadata] using h.dtype.1

/-- Per-coordinate numerical transport uses three local scalar domains. Those
domains do not narrow Python admission or the unconditional execution/frame result. -/
theorem checked_reference_if_local {M : Type} (ops : MemoryPrimitives M FPProof.Word)
    (native reference : Numerics) (domain : Domain) (ports : LocalPorts native reference domain)
    (x y : Tensor) (lx : TensorLaws x) (ly : TensorLaws y)
    (h : InputContract x.metadata y.metadata) (before : M)
    (allocation : EmptyLikeResult ops x y h before)
    (memory : Native.LocalMemory ops (layout ops x y lx ly h before allocation))
    (schedule : Index.Schedule x.metadata.numel) :
    ∃ final,runWrapper ops (formula native) x y lx ly h before allocation schedule.programs=
        .ok (some (allocation.output,final)) ∧
      Preserved ops (Native.view (dataView allocation.output allocation.laws allocation.contiguous) 0) final before ∧
      ∀ i,i<x.metadata.numel → Admitted native domain (logicalValue ops before x lx i) (logicalValue ops before y ly i) →
        logicalValue ops final allocation.output allocation.laws i=
          formula reference (logicalValue ops before x lx i) (logicalValue ops before y ly i) := by
  obtain ⟨final,execution,values,frame⟩ := wrapper_correct ops (formula native) x y lx ly h before allocation memory schedule
  exact ⟨final,execution,frame,fun i hi admitted => (values i hi).trans (transport native reference domain ports _ _ admitted)⟩

#axiomsClean checked_native
#axiomsClean checked_reference_if_local
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivCompute.Contiguous.FPEquiv
