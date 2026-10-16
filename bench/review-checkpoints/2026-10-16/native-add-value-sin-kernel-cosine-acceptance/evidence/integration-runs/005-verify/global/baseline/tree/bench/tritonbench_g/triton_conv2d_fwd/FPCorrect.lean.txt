import bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPComplete
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPCleanUndef
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPSoftware
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPShape
namespace VeriTile.Bench.TritonBenchG.TritonConv2dFwd.FPCorrect
open VeriTile.Triton FP.Structural FPValue
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Complete fp16=true convolution source graph for every original symbolic
shape, stride, padding, group and tf32 option, including all three loops and
zero-trip cases. Address-dependent undef remains arbitrary. The original
BIN>0 / full-channel-block domain is included without adding numerical laws.
The dot primitive and native compiler refinement are explicit external bindings. -/
specification full_backend_correctness {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (initial : State α)
    (hw : ∀ i j, FPValue.active p initial i → FPValue.active p initial j →
      FPValue.outputOffset p initial i = FPValue.outputOffset p initial j → i = j) :
    ∃ final, FP.Structural.exec (backend M D) (source p) initial = some final ∧
      final.mem = (written M D p initial).mem ∧
      (∀ i, FPValue.active p initial i →
        final.mem p.Output (FPValue.outputOffset p initial i)=.mk .fp32 (outputValue M D p initial i)) ∧
      (∀ r o, (r≠p.Output ∨ ∀ i, FPValue.active p initial i → o≠FPValue.outputOffset p initial i) →
        final.mem r o=initial.mem r o) := correctness M D p initial hw

/-- Optional old clean-undef model face, still with genuine per-operation
casts/dot/add. No exact real convolution or native zero-padding claim follows.
In particular the masked zero lanes are not removed from the dot. -/
specification clean_undef_backend_correctness {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (initial : State α)
    (hz : ∀ r o, initial.undef .fp32 r o=M.literal (some .fp32) .real 0)
    (hw : ∀ i j, FPValue.active p initial i → FPValue.active p initial j →
      FPValue.outputOffset p initial i = FPValue.outputOffset p initial j → i = j) :
    ∃ final, FP.Structural.exec (backend M D) (source p) initial = some final ∧
      (∀ i, FPValue.active p initial i →
        final.mem p.Output (FPValue.outputOffset p initial i)=.mk .fp32 (zeroPaddingValue M D p initial i)) ∧
      (∀ r o, (r≠p.Output ∨ ∀ i, FPValue.active p initial i → o≠FPValue.outputOffset p initial i) →
        final.mem r o=initial.mem r o) := clean_undef_correctness M D p initial hz hw

/-- Complete software-bit instance: real binary32→binary16 conversion, exact
half widening, rounded fp32 products and an explicit ascending-lane fp32 sum,
then rounded per-block carry addition. Every callback used by the result is
closed; `rest` is arbitrary and cannot influence bitOutput. Native tensor-core
order, fusion, masked-undef behavior and ABI binding remain separate. -/
specification full_software_correctness (rest : Algebra Word) (p : Parameters)
    (initial : State Word)
    (hw : ∀ i j, FPValue.active p initial i → FPValue.active p initial j →
      FPValue.outputOffset p initial i = FPValue.outputOffset p initial j → i = j) :
    ∃ final, FP.Structural.exec (backend (software rest) sequentialDot) (source p) initial = some final ∧
      (∀ i, FPValue.active p initial i →
        final.mem p.Output (FPValue.outputOffset p initial i)=.mk .fp32 (bitOutput p initial i)) ∧
      (∀ r o, (r≠p.Output ∨ ∀ i, FPValue.active p initial i → o≠FPValue.outputOffset p initial i) →
        final.mem r o=initial.mem r o) := software_correctness rest p initial hw

#axiomsClean full_backend_correctness
#axiomsClean clean_undef_backend_correctness
#axiomsClean full_software_correctness
end
end VeriTile.Bench.TritonBenchG.TritonConv2dFwd.FPCorrect
