import bench.tritonbench_g.quantize_global.Proofs.Execution
import bench.tritonbench_g.quantize_global.Proofs.Real
/-! The absmax is a declared kernel input. Its independent scale condition is
nonnegativity and domination of input magnitudes. Torch's computation of that
buffer is an API boundary, not a new wrapper/reduction implementation proof. -/
namespace QuantizeGlobalRepair.KernelCorrectness
open VeriTile VeriTile.Triton FP.Structural QuantizeIntegerStage
noncomputable section

def ScaleCondition (n : Nat) (xs : Nat → ℝ) (maximum : ℝ) : Prop :=
  0 ≤ maximum ∧ ∀ i, i < n → |xs i| ≤ maximum

def Result (p : Parameters) (v : Invocation) (xs : Nat → ℝ) (maximum : ℝ)
    (initial final : State ℝ) : Prop :=
  (∀ i, i < p.n → i ∈ v.indices →
    final.mem p.Out i = .mk .int (Real.quantized (xs i) maximum)) ∧
  (∀ r i, (r ≠ p.Out ∨ p.n ≤ i ∨ i ∉ v.indices) → final.mem r i = initial.mem r i)

/-- Real nearest-even INT8 tensor result, successful execution and exterior
frame, including zero maximum and every masked tail. -/
theorem correct (rest : Algebra ℝ) (p : Parameters) (v : Invocation)
    (xs : Nat → ℝ) (maximum : ℝ) (initial : State ℝ)
    (inputs : Inputs p xs maximum initial) (scale : ScaleCondition p.n xs maximum) :
    ∃ final, executeProgram rest Real.arithmetic p v initial = some final ∧
      Result p v xs maximum initial final := by
  obtain ⟨final, he, hm⟩ := program_result rest Real.arithmetic p xs maximum initial v inputs
  refine ⟨final, he, ?_, ?_⟩
  · intro i hi hv
    rw [hm]
    simp [patch, hi, hv, Real.signed_readback _ _ scale.1 (scale.2 i hi)]
  · intro r i outside
    rw [hm]
    rcases outside with hr | hi | hv
    · simp [patch, hr]
    · simp [patch, Nat.not_lt.mpr hi]
    · simp [patch, hv]

/-- Reuse across a supplied kernel invocation schedule. This contains no
wrapper allocation, host maximum computation or returned Tensor object. -/
theorem grid_correct (rest : Algebra ℝ) (p : Parameters)
    (xs : Nat → ℝ) (maximum : ℝ) (initial : State ℝ)
    (inputs : Inputs p xs maximum initial) (scale : ScaleCondition p.n xs maximum)
    (sep : p.Separated) (trials : Nat → List Config)
    (nonempty : ∀ k, k < segmentCount p.n → trials k ≠ []) :
    ∃ final, execute rest Real.arithmetic p (schedule p.n trials) initial = some final ∧
      (∀ i, i < p.n → final.mem p.Out i = .mk .int (Real.quantized (xs i) maximum)) ∧
      (∀ r i, (r ≠ p.Out ∨ p.n ≤ i) → final.mem r i = initial.mem r i) := by
  obtain ⟨final, he, hm⟩ := whole_grid rest Real.arithmetic p xs maximum trials nonempty sep initial inputs
  refine ⟨final, he, ?_, ?_⟩
  · intro i hi
    rw [hm]
    simp [complete, hi, Real.signed_readback _ _ scale.1 (scale.2 i hi)]
  · intro r i outside
    rw [hm]
    rcases outside with hr | hi
    · simp [complete, hr]
    · simp [complete, Nat.not_lt.mpr hi]

#specNonCircular ScaleCondition avoiding [source, kernel, scalar, executeProgram, execute]
#specNonCircular Result avoiding [source, kernel, scalar, executeProgram, execute]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end QuantizeGlobalRepair.KernelCorrectness
