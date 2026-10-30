import bench.repairs.tritonbench_g.swiglu_backward.Proofs.Host

namespace VeriTile.Bench.Repairs.SwigluBackward
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SwigluBackward.Whole
noncomputable section

/-- Both source return branches, complete autotuning/grid execution, logical
input binding and exterior memory frame under the executable FP32 guards. -/
specification swiglu_backward_checked_correct (argsMetadata : CallMetadata)
    (rt : Runtime ℝ call entry preexisting) (_binding : CallBinding argsMetadata call)
    (storage : ∀ p : Prepared ℝ call entry preexisting, Storage p) :
    VeriTile.Spec.Real (HostSpec formula (checkedRuntime argsMetadata rt storage)
      (executeHost realAlgebra (checkedRuntime argsMetadata rt storage)) ∧
      ∀ p, (checkedRuntime argsMetadata rt storage).prepare = .ok p → checks p = Bool.true → CheckedIndices p (storage p)) := by
  exact ⟨real_host_outcomes _ (runtimeDomain argsMetadata rt storage),
    fun p hp hc => checked_indices p (storage p) hc (prepared_admitted argsMetadata rt storage p hp hc).2.2⟩

#axiomsClean swiglu_backward_checked_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluBackward
