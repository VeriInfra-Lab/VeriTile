import bench.repairs.tritonbench_g.swiglu_fwd.Proofs.Host

namespace VeriTile.Bench.Repairs.SwigluFwd
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluFwd.Whole
noncomputable section

/-- All raw FP32 payloads, including exceptional words. Cast, sigmoid and
ordered multiplications are explicit primitive ports; no numerical identity
or finite-value premise is used to establish the whole returned tensor. -/
theorem swiglu_fwd_checked_fp32 (M : Algebra (BitVec 32)) (call : CallMetadata)
    (rt : Runtime (BitVec 32) input supplied entry preexisting)
    (_binding : CallBinding call input supplied)
    (storage : ∀ p : Prepared (BitVec 32) input supplied entry preexisting, Storage p) :
    HostSpec (scalar M) (checkedRuntime call rt storage)
      (executeHost M (checkedRuntime call rt storage)) ∧
      ∀ p hp hc, CheckedIndices p (storage p) hc
        (prepared_admitted call rt storage p hp hc).2.2 := by
  exact ⟨checked_host_outcomes M call rt storage,
    fun p hp hc => checked_indices p (storage p) hc (prepared_admitted call rt storage p hp hc).2.2⟩

#axiomsClean swiglu_fwd_checked_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluFwd
