import bench.tritonbench_g.triton_attention.Proofs.FPShape
import bench.tritonbench_g.triton_attention.Proofs.FPWitness
namespace VeriTile.Bench.TritonBenchG.TritonAttention.FPCorrect
open VeriTile.Triton FP.Structural FPValue
open AttentionKernelAligned.FPValue (Plans ascendingPlans)
noncomputable section

/-- Full accepted preprocess source stage: independent initial-memory value
reference, successful execution, readback and complete frame. Floating operations
are explicit backend callbacks; no rounding identity or result premise. -/
specification triton_attention_preprocess_backend_correctness {α : Type} [Inhabited α] (rest : Algebra α) (P : Primitives α) (p : PreParameters) (s : State α)
    (hd : p.newGradient≠p.delta) (hinj : Function.Injective (preOffset p s)) :
    ∃ final, FP.Structural.exec (backend rest P) (preSource p) s=some final ∧
      final.mem=(preWrite P p s s).mem ∧
      (∀ i, final.mem p.newGradient (preOffset p s i)=preGradientCell P p s i) ∧
      (∀ i, final.mem p.delta (preRow p s i)=preDeltaCell P p s i) ∧
      (∀ r o, (r≠p.newGradient ∨ ∀ i, o≠preOffset p s i) →
        (r≠p.delta ∨ ∀ i, o≠preRow p s i) → final.mem r o=s.mem r o) := preprocessCorrectness rest P p s hd hinj

/-- All fp16/bf16/fp32 casts, fp32 division/product and a checked reduction tree are concrete software operations. The external-independent theorem proves the reference cannot depend on any supplied callback in these three profiles. -/
specification triton_attention_preprocess_scalar_bit_correctness [Inhabited Word] (rest : Algebra Word) (external : External) (plans : Plans) (p : PreParameters) (s : State Word) (fmt : StorageFormat) (hfmt : p.dtype=fmt.dtype)
    (hd : p.newGradient≠p.delta) (hinj : Function.Injective (preOffset p s)) :
    ∃ final, FP.Structural.exec (backend rest (software external plans)) (preSource p) s=some final ∧
      final.mem=(preWrite (software external plans) p s s).mem ∧
      (∀ i, final.mem p.newGradient (preOffset p s i)=preGradientCell (software external plans) p s i) ∧
      (∀ i, final.mem p.delta (preRow p s i)=preDeltaCell (software external plans) p s i) ∧
      (∀ r o, (r≠p.newGradient ∨ ∀ i, o≠preOffset p s i) →
        (r≠p.delta ∨ ∀ i, o≠preRow p s i) → final.mem r o=s.mem r o) ∧
      (∀ other : External, final.mem=(preWrite (software other plans) p s s).mem) := by
  obtain ⟨final,he,hm,hg,hd',hf⟩ := preprocessCorrectness rest (software external plans) p s hd hinj
  refine ⟨final,he,hm,hg,hd',hf,?_⟩
  intro other
  rw [hm,preprocess_external_independent external other plans p s fmt hfmt]

#axiomsClean triton_attention_preprocess_backend_correctness
#axiomsClean triton_attention_preprocess_scalar_bit_correctness

/-- Full accepted forward source stage: independent initial-memory value
reference, successful execution, readback and complete frame. Floating operations
are explicit backend callbacks; no rounding identity or result premise. -/
specification triton_attention_forward_backend_correctness {α : Type} [Inhabited α] (rest : Algebra α) (P : Primitives α) (p : ForwardParameters)
    (initial : State α) (hn : 0<p.blockN) (hd : p.blockN ∣ forwardStop p initial)
    (hLO : p.denominator≠p.output) (hMO : p.maximum≠p.output) (hML : p.maximum≠p.denominator)
    (hinj : Function.Injective (forwardOffset p initial)) :
    ∃ final, FP.Structural.exec (backend rest P) (forwardSource p) initial=some final ∧
      final.mem=(forwardWrite P p initial initial).mem ∧
      (∀ i, (forwardOutPointer p initial).inBounds (TileShape.indexToList _ i) [0,1]=Bool.true →
        final.mem p.output (forwardOffset p initial i)=forwardOutput P p initial i) ∧
      (∀ i, final.mem p.denominator (forwardRowOffset p initial i)=forwardLCell P p initial i) ∧
      (∀ i, final.mem p.maximum (forwardRowOffset p initial i)=forwardMCell P p initial i) ∧
      (∀ r o, (r≠p.output ∨ ∀ i, (forwardOutPointer p initial).inBounds (TileShape.indexToList _ i) [0,1]=Bool.true → o≠forwardOffset p initial i) →
        (r≠p.denominator ∨ ∀ i, o≠forwardRowOffset p initial i) →
        (r≠p.maximum ∨ ∀ i, o≠forwardRowOffset p initial i) → final.mem r o=initial.mem r o) := forwardCorrectness rest P p initial hn hd hLO hMO hML hinj

/-- Concrete fp32 scalar arithmetic, fp16/bf16/fp32 casts and checked reduction trees; host scale conversion, exp, typed dot and unsupported f8 casts remain explicit callbacks. This is a backend value contract with partial scalar bit closure, not a fully closed software or native graph theorem. -/
specification triton_attention_forward_scalar_bit_correctness [Inhabited Word] (rest : Algebra Word) (external : External) (plans : Plans) (p : ForwardParameters)
    (initial : State Word) (hn : 0<p.blockN) (hd : p.blockN ∣ forwardStop p initial)
    (hLO : p.denominator≠p.output) (hMO : p.maximum≠p.output) (hML : p.maximum≠p.denominator)
    (hinj : Function.Injective (forwardOffset p initial)) :
    ∃ final, FP.Structural.exec (backend rest (software external plans)) (forwardSource p) initial=some final ∧
      final.mem=(forwardWrite (software external plans) p initial initial).mem ∧
      (∀ i, (forwardOutPointer p initial).inBounds (TileShape.indexToList _ i) [0,1]=Bool.true →
        final.mem p.output (forwardOffset p initial i)=forwardOutput (software external plans) p initial i) ∧
      (∀ i, final.mem p.denominator (forwardRowOffset p initial i)=forwardLCell (software external plans) p initial i) ∧
      (∀ i, final.mem p.maximum (forwardRowOffset p initial i)=forwardMCell (software external plans) p initial i) ∧
      (∀ r o, (r≠p.output ∨ ∀ i, (forwardOutPointer p initial).inBounds (TileShape.indexToList _ i) [0,1]=Bool.true → o≠forwardOffset p initial i) →
        (r≠p.denominator ∨ ∀ i, o≠forwardRowOffset p initial i) →
        (r≠p.maximum ∨ ∀ i, o≠forwardRowOffset p initial i) → final.mem r o=initial.mem r o) := forwardCorrectness rest (software external plans) p initial hn hd hLO hMO hML hinj

#axiomsClean triton_attention_forward_backend_correctness
#axiomsClean triton_attention_forward_scalar_bit_correctness

/-- Full accepted backward source stage: independent initial-memory value
reference, successful execution, readback and complete frame. Floating operations
are explicit backend callbacks; no rounding identity or result premise. -/
specification triton_attention_backward_backend_correctness {α : Type} [Inhabited α] (rest : Algebra α) (P : Primitives α) (p : BackwardParameters)
    (initial : State α) (hBM : 0<p.blockM) :
    ∃ final, FP.Structural.exec (backend rest P) (backwardSource p) initial=some final ∧
      final.mem=backwardFinalMemory P p initial ∧
      (∀ c<p.blocks, ∀ i : BLane p,
        final.mem p.dq (backwardOffset p initial c i)=backwardFinalMemory P p initial p.dq (backwardOffset p initial c i) ∧
        final.mem p.dk (backwardOffset p initial c i)=backwardFinalMemory P p initial p.dk (backwardOffset p initial c i) ∧
        final.mem p.dv (backwardOffset p initial c i)=backwardFinalMemory P p initial p.dv (backwardOffset p initial c i)) ∧
      (∀ r o, ¬backwardWritten p initial r o → final.mem r o=initial.mem r o) := backwardCorrectness rest P p initial hBM

/-- Concrete fp32 scalar arithmetic, fp16/bf16/fp32 casts and checked reduction trees; host scale conversion, exp, typed dot and unsupported f8 casts remain explicit callbacks. This is a backend value contract with partial scalar bit closure, not a fully closed software or native graph theorem. -/
specification triton_attention_backward_scalar_bit_correctness [Inhabited Word] (rest : Algebra Word) (external : External) (plans : Plans) (p : BackwardParameters)
    (initial : State Word) (hBM : 0<p.blockM) :
    ∃ final, FP.Structural.exec (backend rest (software external plans)) (backwardSource p) initial=some final ∧
      final.mem=backwardFinalMemory (software external plans) p initial ∧
      (∀ c<p.blocks, ∀ i : BLane p,
        final.mem p.dq (backwardOffset p initial c i)=backwardFinalMemory (software external plans) p initial p.dq (backwardOffset p initial c i) ∧
        final.mem p.dk (backwardOffset p initial c i)=backwardFinalMemory (software external plans) p initial p.dk (backwardOffset p initial c i) ∧
        final.mem p.dv (backwardOffset p initial c i)=backwardFinalMemory (software external plans) p initial p.dv (backwardOffset p initial c i)) ∧
      (∀ r o, ¬backwardWritten p initial r o → final.mem r o=initial.mem r o) := backwardCorrectness rest (software external plans) p initial hBM

#axiomsClean triton_attention_backward_backend_correctness
#axiomsClean triton_attention_backward_scalar_bit_correctness

local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

theorem triton_attention_preprocess_bit_success (rest : Algebra Word) :
    ∃ final, FP.Structural.exec (backend rest (software witnessExternal ascendingPlans))
      (preSource witnessPre) witnessPreState=some final ∧
      final.mem "NewDO" 0=.mk .fp16 (⟨0x4000⟩ : Word) ∧
      final.mem "NewDO" 1=.mk .fp16 (⟨0x4200⟩ : Word) ∧
      final.mem "Delta" 0=.mk .fp32 (⟨0x40a00000⟩ : Word) ∧
      final.mem "other" 7=.mk .nat 73 := FPValue.preprocess_bit_success rest

#axiomsClean triton_attention_preprocess_bit_success

theorem triton_attention_forward_bit_success (rest : Algebra Word) :
    ∃ final, FP.Structural.exec (backend rest (software witnessExternal ascendingPlans))
      (forwardSource witnessForward) witnessForwardState=some final ∧
      final.mem "Out" 1=.mk .fp16 (⟨0x3e00⟩ : Word) ∧
      final.mem "L" 1=.mk .fp32 (⟨0x40000000⟩ : Word) ∧
      final.mem "M" 1=.mk .fp32 (⟨0⟩ : Word) ∧
      final.mem "other" 7=.mk .nat 73 := FPValue.forward_bit_success rest

#axiomsClean triton_attention_forward_bit_success

theorem triton_attention_backward_bit_success (rest : Algebra Word) :
    ∃ final, FP.Structural.exec (backend rest (software witnessExternal ascendingPlans))
      (backwardSource witnessBackward) witnessBackwardState=some final ∧
      final.mem "DQ" 0=.mk .fp32 (⟨0⟩ : Word) ∧
      final.mem "DQ" 1=.mk .fp32 (⟨0⟩ : Word) ∧
      final.mem "DK" 0=.mk .fp16 (⟨0⟩ : Word) ∧
      final.mem "DV" 0=.mk .fp16 (⟨0x4000⟩ : Word) ∧
      final.mem "DV" 1=.mk .fp16 (⟨0x3c00⟩ : Word) ∧
      final.mem "other" 7=.mk .nat 73 := FPValue.backward_bit_success rest

#axiomsClean triton_attention_backward_bit_success

end
end VeriTile.Bench.TritonBenchG.TritonAttention.FPCorrect
