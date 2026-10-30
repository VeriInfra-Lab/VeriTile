import bench.repairs.tritonbench_g.matmul_tma_static_checked_v2.KernelOnly.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.SourcePreservation
open VeriTile.Triton FPValue

/-- A type-changing constexpr puts the final store in each continuation.
All eight Python statements remain represented by these seven top-level nodes. -/
theorem original_statement_count (p : Parameters) (half : Bool) :
    (source p half).surfaceBody.length = 7 := rfl

theorem candidate_statement_count (p : Parameters) (half : Bool) :
    (sourceExplicitOutput p half).surfaceBody.length = 7 := rfl

private def branchesAtEnd (k : ComputeKernel) : Bool :=
  match k.surfaceBody.reverse.head? with
  | some (.ifThenElse _ _ _) => Bool.true
  | _ => Bool.false

/-- The old typed body combined dot and cast into a single assignment and
had no conditional output conversion. Concrete semantic connections are in
SourceModels; an equality of the two complete ASTs would be false. -/
theorem original_not_historical (p : Parameters) (half : Bool) :
    source p half ≠ FPValue.source p half := by
  intro equal
  have branch := congrArg branchesAtEnd equal
  change Bool.true = Bool.false at branch
  contradiction

#axiomsClean original_statement_count
#axiomsClean candidate_statement_count
#axiomsClean original_not_historical
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.SourcePreservation
