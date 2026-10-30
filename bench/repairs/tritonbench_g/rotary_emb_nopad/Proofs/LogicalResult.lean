import bench.repairs.tritonbench_g.rotary_emb_nopad.IO
import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.RealInterface

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.RotationIO
open VeriTile.Triton FP.Structural KernelInterface
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

theorem logical_uncached_iff (rest : Algebra ℝ) (g : Geometry) (p q : Nat)
    (f : SplitInputs ℝ (ULane g) → Bool → Bool → ULane g → ℝ)
    (formula : f = fun xs key odd i =>
      let a := xs.payload key Bool.false i
      let b := xs.payload key Bool.true i
      let c := xs.cosine i
      let z := xs.sine i
      if odd then a*z+b*c else a*c-b*z) :
    (logicalUncachedIO rest realOps g p q ⊨ f) ↔
      (uncachedIO rest realOps g p q).Implements (uTarget realOps g) := by
  have connection : (logicalUncachedIO rest realOps g p q).lift f = uTarget realOps g := by
    rw [formula]
    funext raw channel
    fin_cases channel <;> rfl
  change (uncachedIO rest realOps g p q).Implements _ ↔ _
  rw [connection]

/-- The signed metadata domain, early-return masks, cache and K frames are
exactly those of the complete previous cached invocation judgment. -/
theorem logical_cached_iff (rest : Algebra ℝ) (g : Geometry) (p q : Nat)
    (f : CachedInputs ℝ (CLane g) → Bool → Bool → CLane g → ℝ)
    (formula : f = fun xs key odd i =>
      let a := xs.payload key Bool.false i
      let b := xs.payload key Bool.true i
      let c := xs.cosine i
      let z := xs.sine i
      if odd then a*z+b*c else a*c-b*z) :
    (logicalCachedIO rest realOps g p q ⊨{LogicalMetadataDomain} f) ↔
      (cachedIO rest realOps g p q).ImplementsOn (MetadataDomain g) (cTarget realOps g) := by
  have connection : (logicalCachedIO rest realOps g p q).lift f = cTarget realOps g := by
    rw [formula]
    funext raw channel
    fin_cases channel <;> rfl
  change (cachedIO rest realOps g p q).ImplementsOn _ _ ↔ _
  rw [connection]
  rfl

#axiomsClean logical_uncached_iff
#axiomsClean logical_cached_iff
end
end VeriTile.Bench.Repairs.RotaryEmbNopad.RotationIO
