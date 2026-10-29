import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.RawKernel
import bench.tritonbench_g.relu_strided_buffer.Proofs.WholeReal

namespace VeriTile.Bench.Repairs.ReluStridedBuffer.KernelRealResult
open VeriTile.Triton FP.Structural
open TritonBenchG.ReluStridedBuffer HostPlan SignedView Whole
noncomputable section

theorem source_scalar (rest : Algebra ℝ) (z : ℝ) :
    scalarResult (realBackend rest) (fun x y => decide (x < y)) z = max 0 z :=
  (scalarResult_eq_historical _ _ (by rfl) z).trans (real_scalar rest z)

/-- Complete Real kernel grid over raw signed pointers: all outputs, input and
exterior frames, and active bounds. The two original branches are retained. -/
theorem relu_strided_buffer_kernel_correctness (rest : Algebra ℝ) (x y : SignedView.View)
    (n inputCapacity outputCapacity : Nat) (hn : 0<n)
    (hin : x.Valid n inputCapacity) (hout : y.Valid n outputCapacity) (hstride : y.stride≠0)
    (hsep : SignedGrid.FootprintsDisjoint x y n) (s : State ℝ) (xs : Nat → ℝ)
    (values : ∀ i, i < n → s.mem x.region (x.address i) = .mk .fp32 (xs i)) :
    VeriTile.Spec.Real (∃ final,
      runGrid (realBackend rest) (Raw.arguments x y n) s = some final ∧
      (∀ i, i < n → final.mem y.region (y.address i) =
        Cell.mk .fp32 (max 0 (xs i))) ∧
      (∀ i, i < n → final.mem x.region (x.address i) =
        s.mem x.region (x.address i)) ∧
      (∀ r a, (r ≠ y.region ∨ ∀ i, i < n → a ≠ y.address i) →
        final.mem r a = s.mem r a) ∧
      (∀ i, i < n → x.address i < inputCapacity ∧
        y.address i < outputCapacity)) := by
  obtain ⟨final,ran,result,inputFrame,frame,bounds⟩ := Raw.grid (realBackend rest)
    (fun x y => decide (x < y)) (real_compare rest) x y hn hin hout hstride hsep s
  refine ⟨final,ran,?_,inputFrame,frame,bounds⟩
  intro i hi
  rw [result i hi,values i hi]
  simp [Cell.read,source_scalar]

#axiomsClean source_scalar
#axiomsClean relu_strided_buffer_kernel_correctness
end
end VeriTile.Bench.Repairs.ReluStridedBuffer.KernelRealResult
