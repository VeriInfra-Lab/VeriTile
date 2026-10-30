import bench.tritonbench_g.context_attn_fwd.IO
import bench.tritonbench_g.context_attn_fwd.Proofs.RealIOComplete

namespace VeriTile.Bench.Repairs.ContextAttnFwd.RealInterface.Logical
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnFwd
open scoped VeriTile.Triton.LogicalDomainIO
noncomputable section

/-- Exact equivalence with the complete old judgment, for every logical
formula, including the unchanged original domain. -/
theorem full_judgment_iff (p : Layout) (r : Regions) (extent : RegionBounds) (g : Grid) (scale : ℝ) (f : Inputs p r extent → Event p g → ℝ) :
    (io p r extent g scale ⊨{domain p r extent g scale} f) ↔
      (RealInterface.io p r extent g scale).ImplementsOn (Domain p r extent g)
        (fun raw _ j => f (inputs p r extent raw) ((Fintype.equivFin (Event p g)).symm j)) := by
  change (RealInterface.io p r extent g scale).ImplementsOn
    (fun raw => Domain p r extent g ((inputs p r extent).symm (inputs p r extent raw)))
    (fun raw _ j => f (inputs p r extent raw) ((Fintype.equivFin (Event p g)).symm j)) ↔ _
  simp only [Equiv.symm_apply_apply]

/-- Mathematical-array presentation; the source/grid/domain judgment is
transported in full. The scalar matching hypothesis concerns pure arrays. -/
theorem correct_arrays (p : Layout) (r : Regions) (extent : RegionBounds) (g : Grid) (scale : ℝ)
    (positive : 0<p.BN) (F : Arrays → Event p g → ℝ)
    (formula_eq : ∀ (a : Formula.Inputs) (e : Event p g),
      F (mathematicalArrays p a) e = RealIO.closedExpected p scale (launch e.1) a e.2) :
    io p r extent g scale ⊨{domain p r extent g scale} (fun xs => F xs.arrays) := by
  apply (full_judgment_iff p r extent g scale _).mpr
  have target_eq : (fun raw (_ : Fin 1) j =>
      F (inputs p r extent raw).arrays ((Fintype.equivFin (Event p g)).symm j)) =
      RealInterface.formula p r extent g scale := by
    funext raw channel j
    exact formula_eq _ _
  exact (congrArg ((RealInterface.io p r extent g scale).ImplementsOn
    (Domain p r extent g)) target_eq).mpr (RealIOComplete.complete p r extent g scale positive)

#axiomsClean full_judgment_iff
end
end VeriTile.Bench.Repairs.ContextAttnFwd.RealInterface.Logical
