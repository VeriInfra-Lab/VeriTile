import bench.repairs.tritonbench_g.decay_cumsum.Proofs.LegacySurface
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.DecayCumsum.SourcePreservation
/-- The legacy hand-written bodies are historical proof artifacts. Separate
program-id statements, signed bounds and the absence of invented store casts
make the actual pasted programs different; no old AST identity is claimed. -/
theorem source_counts (h : Call) (p : Phase) (scale : ℝ) :
    (Kernels.source .forward h p scale).surfaceBody.length=8 ∧
    (Kernels.source .prepare h p scale).surfaceBody.length=11 ∧
    (Kernels.source .backward h p scale).surfaceBody.length=15 := ⟨rfl,rfl,rfl⟩
theorem candidate_counts (h : Call) (p : Phase) (scale : ℝ) :
    (Kernels.indexCommuted .forward h p scale).surfaceBody.length=8 ∧
    (Kernels.indexCommuted .prepare h p scale).surfaceBody.length=11 ∧
    (Kernels.indexCommuted .backward h p scale).surfaceBody.length=15 := ⟨rfl,rfl,rfl⟩
#axiomsClean source_counts
#axiomsClean candidate_counts
end VeriTile.Bench.Repairs.DecayCumsum.SourcePreservation
