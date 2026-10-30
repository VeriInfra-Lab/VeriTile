import bench.tritonbench_g.context_attn_llama.IO
import bench.tritonbench_g.context_attn_llama.Proofs.RealImplementation

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.RealInterface.Logical
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
open scoped VeriTile.Triton.LogicalDomainIO
noncomputable section

/-- Exact equivalence with the complete old judgment, for every logical
formula, including the unchanged original domain. -/
theorem full_judgment_iff (c : Host.Call) (capacity : RegionBounds) (f : Inputs c capacity → Lane c → ℝ) :
    (io c capacity ⊨{domain c capacity} f) ↔
      (programIO c capacity).ImplementsOn (Metadata c capacity)
        (fun raw _ j => f (inputs c capacity raw) ((lanes c).symm j)) := by
  change (programIO c capacity).ImplementsOn
    (fun raw => Metadata c capacity ((inputs c capacity).symm (inputs c capacity raw)))
    (fun raw _ j => f (inputs c capacity raw) ((lanes c).symm j)) ↔ _
  simp only [Equiv.symm_apply_apply]

theorem correct_arrays (c : Host.Call) (capacity : RegionBounds) (geometry : Geometry c capacity)
    (F : Arrays → Lane c → ℝ)
    (formula_eq : ∀ (a : Formula.Inputs) (k : Lane c),
      F (mathematicalArrays c.layout a) k =
        Formula.expected c.layout (c.launchAt k.1) a (Host.scale c.layout.D) k.2.1.val k.2.2.1.val) :
    io c capacity ⊨{domain c capacity} (fun xs => F xs.arrays) := by
  apply (full_judgment_iff c capacity _).mpr
  have target_eq : (fun raw (_ : Fin 1) j => F (inputs c capacity raw).arrays
      ((lanes c).symm j)) = RealInterface.formula c capacity := by
    funext raw channel j
    exact formula_eq _ _
  exact (congrArg ((programIO c capacity).ImplementsOn
    (Metadata c capacity)) target_eq).mpr (RealInterface.implements c capacity geometry)

#axiomsClean full_judgment_iff
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.RealInterface.Logical
