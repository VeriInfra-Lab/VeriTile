import bench.repairs.tritonbench_g.swiglu_backward.Proofs.Host
import bench.tritonbench_g.swiglu_backward.Proofs.Observed

namespace VeriTile.Bench.Repairs.SwigluBackward
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SwigluBackward
open Whole
noncomputable section

/-- Raw FP32 words use the retained observed source graph: the affine factor
has its native FMA, sigmoid and casts are explicit local primitive ports, and
multiply operand order is preserved. No rounded-operation identity is assumed. -/
specification swiglu_backward_checked_fp32 (M : Observed.NativeModel (BitVec 32))
    (argsMetadata : CallMetadata)
    (rt : Runtime (Observed.Term (BitVec 32)) call (Observed.entryGraph entry) preexisting)
    (_binding : CallBinding argsMetadata call)
    (storage : ∀ p : Prepared (Observed.Term (BitVec 32)) call (Observed.entryGraph entry) preexisting, Storage p) :
    Observed.HostSpec M (checkedRuntime argsMetadata rt storage)
      (Observed.executeHost M (checkedRuntime argsMetadata rt storage)) ∧
      ∀ p, (checkedRuntime argsMetadata rt storage).prepare = .ok p → checks p = Bool.true → CheckedIndices p (storage p) := by
  exact ⟨Observed.host_outcomes M _ (runtimeDomain argsMetadata rt storage),
    fun p hp hc => checked_indices p (storage p) hc (prepared_admitted argsMetadata rt storage p hp hc).2.2⟩

#axiomsClean swiglu_backward_checked_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluBackward
