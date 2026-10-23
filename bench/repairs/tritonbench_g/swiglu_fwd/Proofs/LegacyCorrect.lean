import bench.repairs.tritonbench_g.swiglu_fwd.Proofs.Host

namespace VeriTile.Bench.Repairs.SwigluFwd
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SwigluFwd
open VeriTile.Bench.TritonBenchG.SwigluFwd.Whole
noncomputable section

/-- Complete checked FP32-selected entry in the independent Real semantics.
The successful branch includes every output cell and the entry memory frame. -/
theorem swiglu_fwd_checked_correct (call : CallMetadata)
    (rt : Runtime ℝ input supplied entry preexisting) (_binding : CallBinding call input supplied)
    (storage : ∀ p : Prepared ℝ input supplied entry preexisting, Storage p) :
    VeriTile.Spec.Real (HostSpec Math.forward (checkedRuntime call rt storage)
      (executeHost realAlgebra (checkedRuntime call rt storage)) ∧
      ∀ p hp hc, CheckedIndices p (storage p) hc
        (prepared_admitted call rt storage p hp hc).2.2) := by
  exact ⟨real_host_outcomes _ (runtimeDomain call rt storage),
    fun p hp hc => checked_indices p (storage p) hc (prepared_admitted call rt storage p hp hc).2.2⟩

#axiomsClean swiglu_fwd_checked_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluFwd
