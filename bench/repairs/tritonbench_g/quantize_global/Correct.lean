import bench.repairs.tritonbench_g.quantize_global.Proofs.LogicalReader
namespace QuantizeGlobalRepair.Correctness
open VeriTile VeriTile.Triton FP.Structural QuantizeIO
open scoped VeriTile.Triton.LogicalDomainIO
noncomputable section
/-- The supplied logical maximum is nonnegative and dominates input magnitudes.
The kernel does not compute this maximum. -/
def domain {n : Nat} (xs : LogicalView.Inputs n) : Prop :=
  0≤xs.maximum ∧ ∀ i, |xs.values i|≤xs.maximum

specification kernel_grid_correct (rest : Algebra ℝ) (p : Parameters) (sep : p.Separated)
    (trials : Nat → List Config)
    (nonempty : ∀ k, k < segmentCount p.n → trials k ≠ []) :
    Spec.Real (LogicalView.quantizeIO rest p trials ⊨{domain} (fun xs i =>
      if xs.maximum=0 then 0 else QuantizationReal.nearestEven (127*xs.values i/xs.maximum))) :=
  (LogicalReader.judgment_iff rest p trials).mpr
    (ReaderBefore.kernel_grid_correct rest p sep trials nonempty)

#axiomsClean kernel_grid_correct
end
end QuantizeGlobalRepair.Correctness
