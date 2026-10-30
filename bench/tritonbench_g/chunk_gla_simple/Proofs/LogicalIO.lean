import bench.tritonbench_g.chunk_gla_simple.IO
import bench.tritonbench_g.chunk_gla_simple.Proofs.IOReal

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.PublicIO.Logical
open VeriTile Triton Host
open scoped VeriTile.Triton.LogicalScheduledKernelIO
noncomputable section

/-- Exact whole scheduled judgment after bijective tensor coordinates. -/
theorem full_judgment_iff (c : Shape) (scale : ℝ)
    (f : Inputs c → (Fin c.heads × Fin c.T × Fin c.V) → ℝ) :
    (io c scale ⊨ f) ↔ (realIO c scale).RegionalScheduledImplements
      (fun _ => realSchedule c) (fun raw _ j => f (inputs c raw)
        ((coords3 c.heads c.T c.V).symm j)) := Iff.rfl

theorem correct (c : Shape) (bt : 0<c.BT) (scale : ℝ) :
    io c scale ⊨ (fun values coordinate => Host.formula c bt
      (arrays c scale values) scale coordinate.1 coordinate.2.1 coordinate.2.2) := by
  apply (full_judgment_iff c scale _).mpr
  have target_eq : (fun raw (_ : Fin 1) j => Host.formula c bt
      (arrays c scale (inputs c raw)) scale
        ((coords3 c.heads c.T c.V).symm j).1
        ((coords3 c.heads c.T c.V).symm j).2.1
        ((coords3 c.heads c.T c.V).symm j).2.2) = PublicMath.formula c bt scale := by
    funext raw channel j
    unfold arrays
    rw [Equiv.symm_apply_apply]
    rfl
  exact (congrArg ((realIO c scale).RegionalScheduledImplements
    (fun _ => realSchedule c)) target_eq).mpr (real_implements c bt scale)

#axiomsClean full_judgment_iff
end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.PublicIO.Logical
