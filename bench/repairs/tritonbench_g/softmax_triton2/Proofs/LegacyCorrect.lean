import bench.repairs.tritonbench_g.softmax_triton2.Kernels
import bench.repairs.tritonbench_g.softmax_triton2.Contract
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.RealWrapper
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.Host

namespace VeriTile.Bench.Repairs.SoftmaxTriton2.LegacyCorrect
open VeriTile.Bench.Repairs.SoftmaxTriton1
open VeriTile.Triton Contiguous
noncomputable section

/-- Independent mathematical softmax, with no block-size dependency. -/
def reference (n _B : Nat) (xs : Nat → ℝ) (c : Nat) : ℝ :=
  Real.exp (xs c) / ∑ j ∈ Finset.range n, Real.exp (xs j)

theorem reference_congr (n B : Nat) (xs ys : Nat → ℝ) (c : Nat)
    (hc : c<n) (same : ∀ j,j<n → xs j=ys j) : reference n B xs c=reference n B ys c := by
  unfold reference
  rw [same c hc]
  congr 1
  exact Finset.sum_congr rfl (fun j hj => congrArg Real.exp (same j (Finset.mem_range.mp hj)))

specification python_checks_iff_preconditions (x : ArgumentMetadata) :
    checkInput x=.ok () ↔ AdmittedArgument x := checkInput_iff x

specification softmax_wrapper_correct (layout : Layout) (x : Object) (h : InputContract x.metadata)
    (before : World BlockState) (valid : Valid layout before (descriptor x))
    (allocation : Allocation RealModel.access layout x before) :
    ∃ final,runWrapper RealModel.access layout RealModel.evaluate x before allocation=.ok allocation.output final ∧
      Returned RealModel.access Neg.neg layout reference (descriptor x) (descriptor allocation.output) before final ∧
      (∀ r c,r<rows x.metadata → c<cols x.metadata →
        objectValue RealModel.access Neg.neg final.program allocation.output r c=
          reference (cols x.metadata) (blockSize (cols x.metadata))
            (objectValue RealModel.access Neg.neg before.program x r) c) :=
  returned_tensor RealModel.access Neg.neg layout RealModel.evaluate reference RealModel.source
    reference_congr RealModel.source_terminates x h before valid allocation

#specNonCircular reference avoiding [softmax_kernel,runWrapper,RealModel.evaluate]
#axiomsClean python_checks_iff_preconditions
#axiomsClean softmax_wrapper_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxTriton2.LegacyCorrect
