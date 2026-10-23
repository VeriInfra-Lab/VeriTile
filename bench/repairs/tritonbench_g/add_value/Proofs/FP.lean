import bench.repairs.tritonbench_g.add_value.Proofs.Host
import VeriTile.Triton.Float.ScalarOps

namespace VeriTile.Bench.Repairs.AddValue.Contiguous.FPProof
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AddExample.Finite
noncomputable section

abbrev Word := FP.Value .fp32

theorem scalar_exact : scalarWord.toRat? = some 10 := by decide +kernel

/-- Mathematical projection only; the FP rewrite acts on originalKernel itself. -/
theorem same_source (n : Nat) :
    (originalKernel n).toAlgKernel =
      (VeriTile.Bench.TritonBenchG.AddValue.puzzle1_kernel "x" "out" n blockSize 10).toAlgKernel :=
  Projection.kernel_algorithm "x" "out" n blockSize

/-- The finite physical interpreter checks the shared source's explicit
FP32 instruction tags before selecting its existing Word-memory/add10 ports. -/
theorem typed_ast_lowering (n : Nat) :
    Source.lowerComputeBody (originalKernel n).surfaceBody =
      some (Source.sourceCode "x" "out" n) :=
  Source.original_ast_lowering "x" "out" n

/-- Optional model-conformance target: IEEE RNE, no input or result FTZ.
This software expression is not a numerical-transformation experiment goal. -/
def reference (x : Word) : Word := FP.add FP.Config.ieeeValue x scalarWord

/-- Optional model-conformance hypothesis for the software target, separate
from ADD-COMMUTE and its accepted transformation report. This is not an
experiment TODO. NaN/Inf remain in the unconditional structural domain. -/
def LocalAdd10 (nativeAdd : Word → Word → Word) : Prop :=
  ∀ x, x.toRat?.isSome → nativeAdd x scalarWord = reference x

/-- Assemble the complete structural result from local execution, pointer,
allocation, returned-Tensor and exterior-frame facts. -/
theorem tensor_correct {M : Type} (ops : MemoryPrimitives M Word)
    (nativeAdd : Word → Word → Word) (x : Binding.Tensor)
    (lx : Binding.TensorLaws x) (h : InputContract x.metadata) (before : M)
    (a : Binding.EmptyLikeResult ops x h before)
    (runtime : Binding.Runtime ops x lx h before a)
    (memory : Native.LocalMemory ops (Binding.layout ops x lx h before a))
    (schedule : Index.Schedule x.metadata.numel) :
    ∃ final,
      Binding.runWrapper ops (fun v => nativeAdd v scalarWord) x lx h before a
        schedule.programs = .ok (some (a.output, final)) ∧
      (∀ i, i < x.metadata.numel → Binding.logicalValue ops final a.output a.laws i =
        nativeAdd (Binding.logicalValue ops before x lx i) scalarWord) ∧
      Preserved ops (Native.view (Binding.dataView a.output a.laws a.contiguous) 0) final before ∧
      (0 < x.metadata.numel →
        (Native.view (Binding.dataView x lx h.contiguous) 0).pointer = x.dataPtr ∧
        (Native.view (Binding.dataView a.output a.laws a.contiguous) 0).pointer = a.output.dataPtr) ∧
      ops.launchDevice = x.metadata.device.index ∧
      a.output.metadata.shape = x.metadata.shape ∧
      a.output.metadata.device = x.metadata.device ∧
      a.output.metadata.dtype = .floating .fp32 := by
  obtain ⟨final, he, hv, hf⟩ := Binding.wrapper_correct ops
    (fun v => nativeAdd v scalarWord) x lx h before a memory schedule
  refine ⟨final, he, hv, hf, ?_, runtime.currentDevice, ?_, ?_, ?_⟩
  · intro nonempty
    exact ⟨Binding.actual_pointer x lx h.contiguous nonempty,
      Binding.actual_pointer a.output a.laws a.contiguous
        (by simpa [a.metadata, VeriTile.Torch.TensorMetadata.numel] using nonempty)⟩
  · simp [a.metadata]
  · simp [a.metadata]
  · simpa [a.metadata] using h.dtype

/-- Optional model-conformance consequence of an explicitly supplied IEEE
hypothesis. It is not the task's floating-point transformation goal. -/
theorem ieee_if_local {M : Type} (ops : MemoryPrimitives M Word)
    (nativeAdd : Word → Word → Word) (numerical : LocalAdd10 nativeAdd)
    (x : Binding.Tensor) (lx : Binding.TensorLaws x) (h : InputContract x.metadata)
    (before : M) (a : Binding.EmptyLikeResult ops x h before)
    (memory : Native.LocalMemory ops (Binding.layout ops x lx h before a))
    (schedule : Index.Schedule x.metadata.numel) :
    ∃ final,
      Binding.runWrapper ops (fun v => nativeAdd v scalarWord) x lx h before a
        schedule.programs = .ok (some (a.output, final)) ∧
      ∀ i, i < x.metadata.numel → (Binding.logicalValue ops before x lx i).toRat?.isSome →
        Binding.logicalValue ops final a.output a.laws i =
          reference (Binding.logicalValue ops before x lx i) := by
  obtain ⟨final, he, hv, _⟩ := Binding.wrapper_correct ops
    (fun v => nativeAdd v scalarWord) x lx h before a memory schedule
  exact ⟨final, he, fun i hi finite => (hv i hi).trans (numerical _ finite)⟩

#axiomsClean tensor_correct
#axiomsClean ieee_if_local
#axiomsClean scalar_exact
#axiomsClean same_source
#axiomsClean typed_ast_lowering
#specNonCircular reference avoiding [Source.runProgram,Binding.runWrapper]
#specNonCircular LocalAdd10 avoiding [Source.runProgram,Binding.runWrapper]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AddValue.Contiguous.FPProof
