import bench.repairs.tritonbench_g.multinomial_sampling.Correct
import bench.repairs.tritonbench_g.multinomial_sampling.Misc.NativeCorrect
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.TensorReference

/-! Optional actual-Tensor applications and input/check binding helpers for the
primary mathematical and native kernel certificates. -/
namespace VeriTile.Bench.Repairs.MultinomialSampling.TensorBridge
open VeriTile.Triton FP.Structural SourceValue
noncomputable section

/-- The public Tensor target uses logical scores, labels, seed and offset from
one entry-memory reader. `Spec.sample` performs the source's signed32 offset
narrowing before canonical Philox10 decoding. -/
def TensorResult (p : Parameters) (input : TensorInputs.Inputs p) (bytes : TensorInputs.Bytes)
    (r : Nat) (output : Int) : Prop :=
  Spec.Result p.tokens (TensorInputs.weights p input bytes r)
    (fun j => (TensorInputs.labelWord p input bytes r j).toInt)
    (TensorInputs.seedWord p input bytes r).toInt
    (TensorInputs.offsetWord p input bytes r).toInt output

/-- Tensor-bound successful Real execution. The initial state is bound through
local physical loads; finite FP32 guard readback derives the probability domain
inside the proof. No whole-state/output correctness is a premise. -/
theorem tensor_kernel_correctness (rest : Algebra ℝ) (p : Parameters)
    (input : TensorInputs.Inputs p) (bytes : TensorInputs.Bytes) (s : State ℝ)
    (loaded : TensorInputs.Loaded p input bytes TensorInputs.denote s)
    (observed : TensorInputs.Probabilities p input bytes) (outputBase : Nat)
    (native : KernelMemory.Domain p input.scores.dataPtr.toNat
      input.labels.tensor.metadata.dataPtr.toNat input.seeds.tensor.metadata.dataPtr.toNat
      input.offsets.tensor.metadata.dataPtr.toNat outputBase)
    (launched : s.pids 0<KernelMemory.grid p) :
    VeriTile.Spec.Real (∃ t,
      FP.Structural.exec (backend rest Real.primitives) (kernel p) s=some t ∧
      (∀ i : Row p,active p s i → TensorResult p input bytes (row p s i)
        ((t.mem p.output (row p s i)).read .int)) ∧
      (∀ r a,(r≠p.output ∨ ∀ i : Row p,active p s i → a≠row p s i) → t.mem r a=s.mem r a) ∧
      (∀ i : Row p,∀ position,
        input.labels.tensor.metadata.dataPtr.toNat+8*(row p s i*p.strideIndexBatch+
          min position (p.tokens-1)*p.strideIndexToken)<2^64)) := by
  have positive : 0<p.tokens := by have := native.tokens; omega
  have weights (i : Row p) (ha : active p s i) :
      CDFBridge.weights p s i=TensorInputs.weights p input bytes (row p s i) := by
    funext j
    by_cases hj : j<p.tokens
    · simp only [CDFBridge.weights,TensorInputs.weights,hj,↓reduceIte]
      exact TensorInputs.score_read p input bytes TensorInputs.denote s loaded _ j ha hj
    · simp only [CDFBridge.weights,TensorInputs.weights,hj,↓reduceIte]
  have domains : ∀ i : Row p,active p s i → Spec.Domain p.tokens (CDFBridge.weights p s i) := by
    intro i ha
    rw [weights i ha]
    exact TensorInputs.probability_domain p input bytes observed positive _ ha
  obtain ⟨t,he,hv,hf⟩ := Correct.multinomial_kernel_correctness rest p s
    (TensorInputs.seed_domain p input bytes TensorInputs.denote s loaded) domains
  refine ⟨t,he,?_,hf,?_⟩
  · intro i ha
    obtain ⟨j,hj,hout⟩ := hv i ha
    have hw := weights i ha
    have hseed := TensorInputs.seed_read p input bytes TensorInputs.denote s loaded _ ha
    have hoff := TensorInputs.offset_read p input bytes TensorInputs.denote s loaded _ ha
    refine ⟨j,?_,?_⟩
    · simpa only [hw,hseed,hoff] using hj
    · change _=(TensorInputs.labelWord p input bytes (row p s i) j).toInt
      rw [hout]
      exact TensorInputs.label_read p input bytes TensorInputs.denote s loaded _ j ha hj.1
  · intro i position
    exact (KernelMemory.selected_address p _ _ _ _ _ native s launched i position).2

/-- Successful FP32 execution on the logical values of the actual input
Tensors. All input roles share one physical entry-memory reader. The rounded
recurrence explicitly uses logical words, canonical RNG and signed32 offset
narrowing; it is not an exact-to-Real refinement claim. -/
theorem tensor_kernel_words (rest : Algebra Word) (plans : Plans) (p : Parameters)
    (input : TensorInputs.Inputs p) (bytes : TensorInputs.Bytes) (s : State Word)
    (loaded : TensorInputs.Loaded p input bytes TensorInputs.encode s) (outputBase : Nat)
    (native : KernelMemory.Domain p input.scores.dataPtr.toNat
      input.labels.tensor.metadata.dataPtr.toNat input.seeds.tensor.metadata.dataPtr.toNat
      input.offsets.tensor.metadata.dataPtr.toNat outputBase)
    (launched : s.pids 0<KernelMemory.grid p) :
    let data := TensorReference.logicalData p input bytes TensorInputs.encode (s.pids 0) (s.undef .int)
    ∃ t,FP.Structural.exec (backend rest (software .fp32 plans)) (kernel p) s=some t ∧
      t.mem=(TensorReference.written (software .fp32 plans) p data s).mem ∧
      (∀ i : Row p,active p s i → t.mem p.output (row p s i)=Cell.mk .int
        ((TensorReference.rounds (software .fp32 plans) p data (count p)).output i)) ∧
      (∀ r a,(r≠p.output ∨ ∀ i : Row p,active p s i → a≠row p s i) → t.mem r a=s.mem r a) ∧
      (∀ i : Row p,∀ position,
        input.labels.tensor.metadata.dataPtr.toNat+8*(row p s i*p.strideIndexBatch+
          min position (p.tokens-1)*p.strideIndexToken)<2^64) := by
  have positive : 0<p.tokens := by have := native.tokens; omega
  obtain ⟨t,he,hm,hv,hf⟩ := FPEquiv.exact_kernel_words rest plans p s
    (TensorInputs.seed_domain p input bytes TensorInputs.encode s loaded)
  refine ⟨t,he,?_,?_,hf,?_⟩
  · exact hm.trans (TensorReference.written_eq (software .fp32 plans) p input bytes
      TensorInputs.encode s loaded positive)
  · intro i ha
    rw [hv i ha,TensorReference.rounds_eq (software .fp32 plans) p input bytes
      TensorInputs.encode s loaded positive]
  · intro i position
    exact (KernelMemory.selected_address p _ _ _ _ _ native s launched i position).2

alias native_formed_addresses := KernelMemory.formed_addresses
alias selected_load_capacity := KernelMemory.selected_capacity
alias score_tensor_binding := KernelMemory.source_score
alias integer_tensor_binding := KernelMemory.source_integer
alias probability_guard_domain := KernelMemory.probability_domain

#specNonCircular TensorResult avoiding [kernel,source,FP.Structural.exec,rounds,written]
#axiomsClean tensor_kernel_correctness
#axiomsClean tensor_kernel_words
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MultinomialSampling.TensorBridge
