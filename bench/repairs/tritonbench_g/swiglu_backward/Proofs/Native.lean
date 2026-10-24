import bench.repairs.tritonbench_g.swiglu_backward.Proofs.Physical
import bench.tritonbench_g.swiglu_backward.Proofs.Observed

namespace VeriTile.Bench.Repairs.SwigluBackward.KernelProof
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward
open Whole
noncomputable section

/-- Exact words from the independent ordered native graph. The ports describe
the selected FP32 machine instructions, including their special-value policy;
no arithmetic rewrite, exact-sigmoid premise or finite-only guard is used. -/
def NativeResult (M : Observed.NativeModel (BitVec 32))
    (p : Prepared (Observed.Term (BitVec 32)) call (Observed.entryGraph entry) preexisting)
    (s : Storage p) (final : State (Observed.Term (BitVec 32))) : Prop :=
  (∀ ch, enabled call.recompute ch → ∀ r q, r < p.batchShape.prod → q < p.ncols →
    Observed.interpret M (physicalCommit p s final call.input.cudaDevice (outputAddress p s ch r q)) =
      Observed.reference M ch
        (call.input.read entry (r*(2*p.ncols)+q))
        (call.input.read entry (r*(2*p.ncols)+(p.ncols+q)))
        (call.upstream.read entry (r*p.ncols+q))) ∧
  (∀ device address, address % 4 = 0 → ¬PhysicalWritten p s device address →
    Observed.interpret M (physicalCommit p s final device address) =
      Observed.interpret M (s.preparedRead device address))

theorem native_result (M : Observed.NativeModel (BitVec 32))
    (p : Prepared (Observed.Term (BitVec 32)) call (Observed.entryGraph entry) preexisting)
    (s : Storage p) (final : State (Observed.Term (BitVec 32)))
    (result : PhysicalResult (scalar Observed.graphAlgebra) p s final) : NativeResult M p s final := by
  refine ⟨?_,?_⟩
  · intro ch on r q hr hq
    have value := congrArg (Observed.interpret M) (result.1 ch on r q hr hq)
    simpa only [VeriTile.Bench.TritonBenchG.SwigluFwd.Whole.Tensor.read,
      Observed.entryGraph,Observed.source_graph_lowering] using value
  · intro device address aligned outside
    exact congrArg (Observed.interpret M) (result.2 device address aligned outside)

#axiomsClean native_result
#guard_msgs (drop info) in
#specNonCircular Observed.reference avoiding [Observed.interpret,scalar,run,physicalCommit]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluBackward.KernelProof
