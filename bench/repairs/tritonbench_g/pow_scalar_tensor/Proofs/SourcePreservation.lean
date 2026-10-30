import bench.repairs.tritonbench_g.pow_scalar_tensor.Kernels
import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.HistoricalBody
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked.SourcePreservation
open VeriTile.Triton

theorem original_statement_count (c : Parameters) : (originalBody c).surfaceBody.length = 3 := rfl
theorem candidate_statement_count (c : Parameters) : (commutedBody c).surfaceBody.length = 3 := rfl

/-- The old writer erased the original constexpr branch and block pointers.
It is not an AST identity with the pasted body. Source execution is connected
in Proofs.Source and Proofs.Kernel, retaining the formerly omitted cast. -/
theorem original_not_historical (c : Parameters) (old : HistoricalBody.Parameters) :
    originalBody c ≠ HistoricalBody.oneTileKernel old := by
  intro equal
  have counts := congrArg (fun k : ComputeKernel => k.surfaceBody.length) equal
  change (3 : Nat) = 10 at counts
  contradiction

#axiomsClean original_statement_count
#axiomsClean candidate_statement_count
#axiomsClean original_not_historical
end VeriTile.Bench.Repairs.PowScalarTensor.Checked.SourcePreservation
