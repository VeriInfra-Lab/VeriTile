import bench.tritonbench_g.fifth_order_sph_harmonics.Proofs.IOCorrect

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.LogicalIO
open VeriTile.Triton FP.Structural HarmonicIO KernelScalar
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

theorem forward_iff (p : Parameters) (f : Coordinates p → Harmonics p) :
    (forwardView p ⊨ f) ↔ (forwardIO realBackend p).Implements
      (forwardTarget realBackend p (fun xs i => f (fun r c => xs (r,c,())) i.1 i.2.1)) := by
  have eq : (forwardView p).lift f = forwardTarget realBackend p
      (fun xs i => f (fun r c => xs (r,c,())) i.1 i.2.1) := by
    funext raw port j
    fin_cases port
    rfl
  change (forwardIO realBackend p).Implements _ ↔ _
  rw [eq]

theorem backward_iff (p : Parameters) (f : BackwardInputs p → Coordinates p) :
    (backwardView p ⊨{Prepared p} f) ↔ (backwardIO realBackend p).ImplementsOn
      (prepared p) (backwardTarget realBackend p
        (fun xs gs old i => f ⟨fun r c => xs (r,c,()),fun r c => gs (r,c,()),fun r c => old (r,c,())⟩ i.1 i.2.1)) := by
  have domain : (fun raw => Prepared p ((backwardInputs p) raw)) = prepared p := by
    funext raw
    apply propext
    constructor
    · intro h i; exact h i.1 i.2.1
    · intro h r c; exact h (r,c,())
  change (backwardIO realBackend p).ImplementsOn _ _ ↔ _
  change (backwardIO realBackend p).ImplementsOn (fun raw => Prepared p (backwardInputs p raw)) _ ↔ _
  rw [domain]
  have eq : (backwardView p).lift f = backwardTarget realBackend p
      (fun xs gs old i => f ⟨fun r c => xs (r,c,()),fun r c => gs (r,c,()),fun r c => old (r,c,())⟩ i.1 i.2.1) := by
    funext raw port j
    fin_cases port
    rfl
  rw [eq]
end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.LogicalIO
