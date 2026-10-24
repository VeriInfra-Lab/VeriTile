import bench.repairs.tritonbench_g.quantize_global.Proofs.KernelCorrectness
/-! Kernel-only correctness of the existing separate repair. The passed absmax
is nonnegative and bounds every input magnitude. The mathematical output is
zero for absmax=0, otherwise nearest-even(127*x[i]/absmax), as signed INT8.
The source retains explicit div.rn, separate multiply, clamp and llrint64.
No wrapper, allocator or Torch reduction implementation is a proof target. -/
namespace QuantizeGlobalRepair.KernelRealResults
open VeriTile VeriTile.Triton FP.Structural
noncomputable section

theorem kernel_correct (rest : Algebra ℝ) (p : Parameters) (v : Invocation)
    (xs : Nat → ℝ) (maximum : ℝ) (initial : State ℝ)
    (inputs : Inputs p xs maximum initial) (scale : KernelCorrectness.ScaleCondition p.n xs maximum) :
    Spec.Real (∃ final, executeProgram rest Real.arithmetic p v initial = some final ∧
      KernelCorrectness.Result p v xs maximum initial final) :=
  KernelCorrectness.correct rest p v xs maximum initial inputs scale

theorem kernel_grid_correct (rest : Algebra ℝ) (p : Parameters)
    (xs : Nat → ℝ) (maximum : ℝ) (initial : State ℝ)
    (inputs : Inputs p xs maximum initial) (scale : KernelCorrectness.ScaleCondition p.n xs maximum)
    (sep : p.Separated) (trials : Nat → List Config)
    (nonempty : ∀ k, k < segmentCount p.n → trials k ≠ []) :
    Spec.Real (∃ final, execute rest Real.arithmetic p (schedule p.n trials) initial = some final ∧
      (∀ i, i < p.n → final.mem p.Out i = .mk .int (Real.quantized (xs i) maximum)) ∧
      (∀ r i, (r ≠ p.Out ∨ p.n ≤ i) → final.mem r i = initial.mem r i)) :=
  KernelCorrectness.grid_correct rest p xs maximum initial inputs scale sep trials nonempty

#guard_msgs (drop info) in
#auditModuleAxioms
end
end QuantizeGlobalRepair.KernelRealResults
