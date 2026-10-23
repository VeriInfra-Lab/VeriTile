import bench.repairs.tritonbench_g.quant_transpose_kernel.Correct
import bench.repairs.tritonbench_g.quant_transpose_kernel.FPEquiv
import bench.repairs.tritonbench_g.quant_transpose_kernel.Proofs.KernelTensor

/-! Optional Tensor applications and native input/pointer bindings. -/
namespace VeriTile.Bench.Repairs.QuantTransposeKernel.TensorBridge
open VeriTile.Triton FP.Structural KernelModels KernelInputs
open QuantTransposeFinite (Allocations PhysicalLayout)
noncomputable section

/-- Complete actual grouped kernel under the checked Python contract. Real
is the mathematical interpretation: ordered multiplication, nearest-even
signed64 conversion with explicit range handling, then INT8 observation.
The inverse is the actual kernel input; no host reciprocal theorem is added. -/
theorem kernel_correct (rest : Algebra ℝ) (md : Metadata)
    (_admitted : InputContract md) (alloc : QuantTransposeFinite.Allocations)
    (x : String) (fresh : alloc.Fresh x) (values : Nat → Nat → ℝ)
    (inverse : ℝ) (initial : State ℝ)
    (reads : KernelWhole.Reads (parameters md.input alloc x) values inverse initial) :
    ∃ final,
      KernelGrid.run (realAlgebra rest) realRound (parameters md.input alloc x) Bool.false
        (List.range (QuantTransposeFinite.gridM md.input.M * QuantTransposeFinite.gridN md.input.N))
        initial = some final ∧
      KernelGrid.run (realAlgebra rest) realRound (parameters md.input alloc x) Bool.true
        (List.range (QuantTransposeFinite.gridM md.input.M * QuantTransposeFinite.gridN md.input.N))
        initial = some final ∧
      KernelWhole.Result (parameters md.input alloc x)
        (fun r c => realByte (values r c) inverse) initial final := by
  obtain ⟨final, he, hr⟩ := KernelWhole.real_correct rest (parameters md.input alloc x)
    (separated md.input alloc x fresh) values inverse initial reads
  exact ⟨final, he, (KernelGrid.same_run _ _ _ _ _).symm.trans he, hr⟩

/-- A store observation at the actual output Tensor base. Only the byte
written by the source's int8 pointer is observed; this is not a reconstructed
final byte heap or a proof of allocation/runtime internals. -/
def outputStore (md : Metadata) (alloc : Allocations) (physical : PhysicalLayout)
    (final : State Word) (r c : Nat) : Nat × BitVec 8 :=
  (physical.output.base + c * md.input.M + r,
    KernelWhole.byte final alloc.output (c * md.input.M + r))

/-- Both complete actual launches succeed with the same state. The result
uses actual FP32 inputs and inverse, ordered native multiplies, the pinned
sm80 signed64/FTZ conversion and low8 store. NaNs, infinities and zero inputs
remain admitted. There is no numerical rewrite or successful-output premise. -/
theorem kernel_equiv (multiply : Word → Word → Word) (rest : Algebra Word)
    (md : Metadata) (admitted : InputContract md) (alloc : Allocations)
    (memory : QuantTransposeFinite.Memory) (x : String)
    (logical : List (QuantTransposeFinite.Word .fp32))
    (xs : Nat → QuantTransposeFinite.Word .fp32) (physical : PhysicalLayout)
    (domain : QuantTransposeFinite.Domain .fp32 md.input alloc memory x logical xs)
    (ordinary : OrdinaryPhysicalBinding md.input alloc memory x physical)
    (pointer : physical.inputPointer.toNat = md.dataPointer)
    (inverse : Word) (initial : State Word)
    (reads : NativeReads md.input alloc memory x physical inverse initial) :
    ∃ final,
      KernelGrid.run (nativeAlgebra multiply rest) nativeRound (parameters md.input alloc x) Bool.false
        (List.range (QuantTransposeFinite.gridM md.input.M * QuantTransposeFinite.gridN md.input.N))
        initial = some final ∧
      KernelGrid.run (nativeAlgebra multiply rest) nativeRound (parameters md.input alloc x) Bool.true
        (List.range (QuantTransposeFinite.gridM md.input.M * QuantTransposeFinite.gridN md.input.N))
        initial = some final ∧
      KernelWhole.Result (parameters md.input alloc x)
        (fun r c => nativeByte multiply ((⟨xs (c * md.input.M + r)⟩ : Word)) inverse) initial final ∧
      (∀ r c, r < md.input.M → c < md.input.N →
        outputStore md alloc physical final r c =
          (alloc.outputBase + c * md.input.M + r,
            nativeByte multiply ((⟨xs (c * md.input.M + r)⟩ : Word)) inverse)) ∧
      QuantTransposeFinite.ArithmeticRefines md.input .fp32 physical := by
  obtain ⟨final, he, hr⟩ := KernelWhole.native_correct multiply rest (parameters md.input alloc x)
    (separated md.input alloc x domain.fresh)
    (fun r c => (⟨xs (c * md.input.M + r)⟩ : Word)) inverse initial
    (native_reads md.input alloc memory x logical xs physical inverse initial domain reads)
  refine ⟨final, he, (KernelGrid.same_run _ _ _ _ _).symm.trans he, hr, ?_, ?_⟩
  · intro r c hrow hcol
    simp only [outputStore, ordinary.outputBase]
    exact Prod.ext rfl (hr.1 r c hrow hcol)
  · exact arithmetic_refines md.input admitted.indices .fp32 alloc memory x logical xs physical
      domain (ordinary.admitted admitted pointer)

/-- Exact source-lane addresses, including checked native signed-int32 mask
and pointer arithmetic, reuse the existing arithmetic proof. -/
alias native_pointer_binding := KernelInputs.native_addresses
alias native_input_byte_binding := KernelInputs.input_bytes

/-- Actual shared input Tensor logical values and the prepared scalar Tensor
determine every output Tensor coordinate. Descriptor words are linked by
KernelTensor.words_eq_logical, rather than treated as unrelated arrays.
The ordinary INT8 output binding supplies metadata and physical indexing;
there is no host reciprocal, allocator or final-byteheap theorem. -/
theorem kernel_tensor_equiv (multiply : Word → Word → Word) (rest : Algebra Word)
    (md : Metadata) (admitted : InputContract md) (alloc : Allocations)
    (memory : QuantTransposeFinite.Memory) (x : String)
    (logical : List (QuantTransposeFinite.Word .fp32))
    (xs : Nat → QuantTransposeFinite.Word .fp32) (physical : PhysicalLayout)
    (domain : QuantTransposeFinite.Domain .fp32 md.input alloc memory x logical xs)
    (ordinary : OrdinaryPhysicalBinding md.input alloc memory x physical)
    (pointer : physical.inputPointer.toNat = md.dataPointer)
    (input : KernelTensor.Tensor) (inputLaws : KernelTensor.TensorLaws input)
    (read : Nat → Word) (binding : KernelTensor.InputBinding md input read memory x physical)
    (inverse : KernelTensor.Tensor) (inverseLaws : KernelTensor.TensorLaws inverse)
    (initial : State Word)
    (inputDecode : ∀ o w, memory x o = some (.mk .fp32 w) →
      initial.mem ⟨x⟩ o = .mk .real (FP.Value.mk w))
    (inverseBinding : KernelTensor.InverseBinding inverse read alloc physical initial)
    (out : KernelTensor.OutputTensor) (outputBinding : KernelTensor.OutputBinding md input out physical) :
    ∃ final,
      KernelGrid.run (nativeAlgebra multiply rest) nativeRound (parameters md.input alloc x) Bool.false
        (List.range (QuantTransposeFinite.gridM md.input.M * QuantTransposeFinite.gridN md.input.N))
        initial = some final ∧
      KernelGrid.run (nativeAlgebra multiply rest) nativeRound (parameters md.input alloc x) Bool.true
        (List.range (QuantTransposeFinite.gridM md.input.M * QuantTransposeFinite.gridN md.input.N))
        initial = some final ∧
      KernelWhole.Result (parameters md.input alloc x)
        (fun r c => nativeByte multiply (KernelTensor.values md input inputLaws read r c)
          (KernelTensor.inverseValue inverse inverseLaws read)) initial final ∧
      (∀ r c, r < md.input.M → c < md.input.N →
        outputStore md alloc physical final r c =
          (KernelTensor.outputAddress out (c * md.input.M + r),
            nativeByte multiply (KernelTensor.values md input inputLaws read r c)
              (KernelTensor.inverseValue inverse inverseLaws read))) ∧
      QuantTransposeFinite.ArithmeticRefines md.input .fp32 physical := by
  obtain ⟨final, original, candidate, result, stores, arithmetic⟩ :=
    kernel_equiv multiply rest md admitted alloc memory x logical xs physical domain ordinary pointer
      (KernelTensor.inverseValue inverse inverseLaws read) initial
      (KernelTensor.native_reads md.input alloc memory x physical inverse inverseLaws read initial
        inputDecode inverseBinding)
  have values := KernelTensor.words_eq_logical md admitted input inputLaws read memory alloc x
    physical binding logical xs domain
  refine ⟨final, original, candidate, ⟨?_, result.2⟩, ?_, arithmetic⟩
  · intro r c hr hc
    exact (result.1 r c hr hc).trans
      (congrArg (fun value => nativeByte multiply value
        (KernelTensor.inverseValue inverse inverseLaws read)) (values r c hr hc))
  · intro r c hr hc
    rw [stores r c hr hc, values r c hr hc]
    rw [← ordinary.outputBase,
      ← KernelTensor.output_address md input out physical outputBinding r c hr hc]

alias tensor_input_pointer := KernelTensor.input_address
alias tensor_inverse_pointer := KernelTensor.inverse_pointer

#axiomsClean kernel_correct
#axiomsClean kernel_equiv
#specNonCircular outputStore avoiding [KernelSource.kernel,KernelExecution.exec,KernelGrid.run]
#axiomsClean native_pointer_binding
#axiomsClean native_input_byte_binding
#axiomsClean kernel_tensor_equiv
#axiomsClean tensor_input_pointer
#axiomsClean tensor_inverse_pointer
end
end VeriTile.Bench.Repairs.QuantTransposeKernel.TensorBridge
