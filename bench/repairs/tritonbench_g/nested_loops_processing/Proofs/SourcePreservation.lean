import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.TypedFP
import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.LegacyFP

/-! Complete success/failure and every-cell connection for the actual original
and retained candidate. Internal signed/Nat register types are not equated. -/
namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.Kernels.SourcePreservation
open VeriTile.Triton FP.Structural

theorem original {α : Type} [Inhabited α] (M : Algebra α)
    (count cols : Nat) (s : State α) :
    (exec M (source count cols) s).map (·.mem) =
      (exec M (LegacyPastedSurface.source count cols) s).map (·.mem) := by
  rw [TypedExecution.source_run,TypedExecution.legacy_source_run]
  exact congrArg some (TypedExecution.source_memory count cols s)

theorem candidate {α : Type} [Inhabited α] (M : Algebra α)
    (count cols : Nat) (s : State α) :
    (exec M (commuted count cols) s).map (·.mem) =
      (exec M (LegacyPastedSurface.commuted count cols) s).map (·.mem) := by
  rw [←TypedExecution.source_equivalence,←LegacyEquivalence.source_equivalence]
  exact original M count cols s

#axiomsClean original
#axiomsClean candidate
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.NestedLoopsProcessing.Kernels.SourcePreservation
