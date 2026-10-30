import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.LogicalIO
import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.HeadlinePolynomial

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.ReaderConnections
open VeriTile.Triton FP.Structural HarmonicIO KernelScalar
open scoped BigOperators VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

theorem backward (p : Parameters) (fit : p.columnOffset + 11 ≤ p.outputWidth)
    (J : ℝ → ℝ → ℝ → (Fin 11 → ℝ) → Fin 3 → ℝ)
    (definition : J = fun x y z upstream component =>
      ∑ channel, upstream channel *
    deriv (fun t => Polynomial.value channel (if component=0 then t else x)
      (if component=1 then t else y) (if component=2 then t else z)) (![x,y,z] component)) :
    LogicalIO.backwardView p ⊨{LogicalIO.Prepared p} (fun xs row component =>
      J (xs.coordinates row 0) (xs.coordinates row 1) (xs.coordinates row 2)
        (xs.upstream row) component) := by
  apply (LogicalIO.backward_iff p _).mpr
  have formula : (fun xs gs (_ : TileIndex [p.rows, 3] → ℝ) => HarmonicIO.realBackward (rows := p.rows) xs gs) =
      (fun xs gs (_ : TileIndex [p.rows, 3] → ℝ) i =>
        J (xs (i.1,0,())) (xs (i.1,1,())) (xs (i.1,2,()))
          (fun channel => gs (i.1,channel,())) i.2.1) := by
    funext xs gs old i
    simp only [HarmonicIO.realBackward, Polynomial.vjp, definition,
      HeadlinePolynomial.jacobian_eq_deriv, Polynomial.coordinate]
  rw [← formula]
  exact IOCorrect.real_backward p fit

end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.ReaderConnections
