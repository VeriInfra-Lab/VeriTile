import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.KernelRealResult
import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.KernelNativeResult
import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.Memory

/-! Optional Tensor/check applications; the primary kernel specifications
remain in Correct.lean and FPEquiv.lean, which do not import this bridge. -/
namespace VeriTile.Bench.Repairs.ReluStridedBuffer.TensorBridge
open VeriTile.Triton FP.Structural
open TritonBenchG.ReluStridedBuffer HostPlan SignedView Whole
noncomputable section
local instance : Inhabited (FP.Value .fp32) := ⟨FP.Value.zero .fp32⟩

theorem python_checks_iff_preconditions (x y : ArgumentMetadata) :
    checkInputs x y = .ok () ↔ AdmittedArguments x y := checkInputs_iff x y

/-- Mathematical ReLU of the shared signed FP32 kernel: every logical output,
input/exterior memory frames and active cell bounds, for both source branches.
Python contributes exactly InputContract; ordinary allocation provenance binds
regions to actual checked storage. No wrapper execution is asserted. -/
theorem relu_strided_buffer_kernel_correctness (rest : Algebra ℝ) (x y : ViewMetadata)
    (h : InputContract x y) (api : AllocationAPI x y) (s : State ℝ) (xs : Nat → ℝ)
    (values : ∀ i, i < x.n → s.mem x.buffer.region ((x.buffer.view x.stride).address i) = .mk .fp32 (xs i)) :
    VeriTile.Spec.Real (∃ final,
      runGrid (realBackend rest) (rankOneArguments x.buffer y.buffer x.n x.stride y.stride) s = some final ∧
      (∀ i, i < x.n → final.mem y.buffer.region ((y.buffer.view y.stride).address i) =
        Cell.mk .fp32 (max 0 (xs i))) ∧
      (∀ i, i < x.n → final.mem x.buffer.region ((x.buffer.view x.stride).address i) =
        s.mem x.buffer.region ((x.buffer.view x.stride).address i)) ∧
      (∀ r a, (r ≠ y.buffer.region ∨ ∀ i, i < x.n → a ≠ (y.buffer.view y.stride).address i) →
        final.mem r a = s.mem r a) ∧
      (∀ i, i < x.n → (x.buffer.view x.stride).address i < x.capacity ∧
        (y.buffer.view y.stride).address i < y.capacity)) := by
  have outputValid : (y.buffer.view y.stride).Valid x.n y.capacity :=
    h.shape.symm ▸ ViewContract.valid y h.output
  obtain ⟨final,ran,result,inputFrame,frame,bounds⟩ := grid_result (realBackend rest)
    (fun x y => decide (x < y)) (real_compare rest) x.buffer y.buffer h.input.extent.1
    x.stride y.stride (ViewContract.valid x h.input) outputValid h.outputStride
    (input_output_footprints x y h api) s
  refine ⟨final,ran,?_,inputFrame,frame,bounds⟩
  intro i hi
  rw [result i hi,values i hi]
  simp [Cell.read,real_scalar]

/-- Checked strided-domain specialization: both complete kernel grids execute,
all words have the independent ordered-ReLU result, and input/exterior frames
and active bounds hold. All NaNs and both zero signs select +0; positive
nonzero words are preserved. Inactive lanes impose no operand restriction. -/
theorem relu_strided_buffer_grid_equiv (rest : Algebra (FP.Value .fp32))
    (x y : ViewMetadata) (h : InputContract x y) (api : AllocationAPI x y)
    (s : State (FP.Value .fp32)) (xs : Nat → FP.Value .fp32)
    (values : ∀ i, i < x.n → s.mem x.buffer.region ((x.buffer.view x.stride).address i) = .mk .fp32 (xs i)) :
    ∃ original folded,
      runGrid (fp32Backend rest) (rankOneArguments x.buffer y.buffer x.n x.stride y.stride) s = some original ∧
      runFoldedGrid (fp32Backend rest) (rankOneArguments x.buffer y.buffer x.n x.stride y.stride) s = some folded ∧
      original.mem = folded.mem ∧
      (∀ i, i < x.n → original.mem y.buffer.region ((y.buffer.view y.stride).address i) =
        Cell.mk .fp32 (fp32Result xs i)) ∧
      (∀ i, i < x.n → original.mem x.buffer.region ((x.buffer.view x.stride).address i) =
        s.mem x.buffer.region ((x.buffer.view x.stride).address i)) ∧
      (∀ r a, (r ≠ y.buffer.region ∨ ∀ i, i < x.n → a ≠ (y.buffer.view y.stride).address i) →
        original.mem r a = s.mem r a) ∧
      (∀ i, i < x.n → (x.buffer.view x.stride).address i < x.capacity ∧
        (y.buffer.view y.stride).address i < y.capacity) := by
  have outputValid : (y.buffer.view y.stride).Valid x.n y.capacity :=
    h.shape.symm ▸ ViewContract.valid y h.output
  obtain ⟨final,ran,result,inputFrame,frame,bounds⟩ := grid_result (fp32Backend rest)
    WordTile.orderedLt (WordTile.backend_compare (zeroLiterals rest) .ieeeValue)
    x.buffer y.buffer h.input.extent.1 x.stride y.stride (ViewContract.valid x h.input)
    outputValid h.outputStride (input_output_footprints x y h api) s
  refine ⟨final,final,ran,(grid_folded_execution _ _ s).symm.trans ran,rfl,?_,inputFrame,frame,bounds⟩
  intro i hi
  rw [result i hi,values i hi]
  simp [Cell.read,fp32Backend,
    WordTile.scalarResult_word _ _ (zero_literal_exact rest),fp32_result_classification,fp32Result]

#axiomsClean relu_strided_buffer_kernel_correctness
#axiomsClean relu_strided_buffer_grid_equiv
#axiomsClean python_checks_iff_preconditions
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ReluStridedBuffer.TensorBridge
