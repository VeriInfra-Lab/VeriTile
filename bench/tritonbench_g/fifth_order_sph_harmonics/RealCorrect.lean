import bench.tritonbench_g.fifth_order_sph_harmonics.Proofs.ReaderConnections
import bench.tritonbench_g.fifth_order_sph_harmonics.IO
import bench.tritonbench_g.fifth_order_sph_harmonics.Proofs.LogicalIO
import bench.tritonbench_g.fifth_order_sph_harmonics.Proofs.HeadlinePolynomial

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Correct
open VeriTile.Triton FP.Structural HarmonicIO KernelScalar
open scoped BigOperators
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Exact decimal-coefficient degree-five polynomials, rather than an
assertion of ideal square-root harmonic normalization. The equality connection
below reuses the canonical independent table. -/
def harmonic (channel : Fin 11) (x y z : ℝ) : ℝ :=
  match channel.val with
  | 0 => 0 + (232681380862329 / 100000000000000 : ℝ) * x^5 * y^0 * z^0 + (-232681380862329 / 10000000000000 : ℝ) * x^3 * y^0 * z^2 + (29085172607791 / 2500000000000 : ℝ) * x^1 * y^0 * z^4
  | 1 => 0 + (-294321253055229 / 10000000000000 : ℝ) * x^3 * y^1 * z^1 + (294321253055229 / 10000000000000 : ℝ) * x^1 * y^1 * z^3
  | 2 => 0 + (34686092313779 / 20000000000000 : ℝ) * x^5 * y^0 * z^0 + (-34686092313779 / 2500000000000 : ℝ) * x^3 * y^2 * z^0 + (-34686092313779 / 10000000000000 : ℝ) * x^3 * y^0 * z^2 + (104058276941337 / 2500000000000 : ℝ) * x^1 * y^2 * z^2 + (-104058276941337 / 20000000000000 : ℝ) * x^1 * y^0 * z^4
  | 3 => 0 + (-10620403417479 / 625000000000 : ℝ) * x^3 * y^1 * z^1 + (339852909359329 / 10000000000000 : ℝ) * x^1 * y^3 * z^1 + (-10620403417479 / 625000000000 : ℝ) * x^1 * y^1 * z^3
  | 4 => 0 + (80282703616657 / 50000000000000 : ℝ) * x^5 * y^0 * z^0 + (-192678488679977 / 10000000000000 : ℝ) * x^3 * y^2 * z^0 + (80282703616657 / 25000000000000 : ℝ) * x^3 * y^0 * z^2 + (128452325786651 / 10000000000000 : ℝ) * x^1 * y^4 * z^0 + (-192678488679977 / 10000000000000 : ℝ) * x^1 * y^2 * z^2 + (80282703616657 / 50000000000000 : ℝ) * x^1 * y^0 * z^4
  | 5 => 0 + (621867148191637 / 100000000000000 : ℝ) * x^4 * y^1 * z^0 + (-16583123951777 / 1000000000000 : ℝ) * x^2 * y^3 * z^0 + (124373429638327 / 10000000000000 : ℝ) * x^2 * y^1 * z^2 + (16583123951777 / 5000000000000 : ℝ) * x^0 * y^5 * z^0 + (-16583123951777 / 1000000000000 : ℝ) * x^0 * y^3 * z^2 + (621867148191637 / 100000000000000 : ℝ) * x^0 * y^1 * z^4
  | 6 => 0 + (80282703616657 / 50000000000000 : ℝ) * x^4 * y^0 * z^1 + (-192678488679977 / 10000000000000 : ℝ) * x^2 * y^2 * z^1 + (80282703616657 / 25000000000000 : ℝ) * x^2 * y^0 * z^3 + (128452325786651 / 10000000000000 : ℝ) * x^0 * y^4 * z^1 + (-192678488679977 / 10000000000000 : ℝ) * x^0 * y^2 * z^3 + (80282703616657 / 50000000000000 : ℝ) * x^0 * y^0 * z^5
  | 7 => 0 + (849632273398321 / 100000000000000 : ℝ) * x^4 * y^1 * z^0 + (-10620403417479 / 625000000000 : ℝ) * x^2 * y^3 * z^0 + (10620403417479 / 625000000000 : ℝ) * x^0 * y^3 * z^2 + (-849632273398321 / 100000000000000 : ℝ) * x^0 * y^1 * z^4
  | 8 => 0 + (104058276941337 / 20000000000000 : ℝ) * x^4 * y^0 * z^1 + (-104058276941337 / 2500000000000 : ℝ) * x^2 * y^2 * z^1 + (34686092313779 / 10000000000000 : ℝ) * x^2 * y^0 * z^3 + (34686092313779 / 2500000000000 : ℝ) * x^0 * y^2 * z^3 + (-34686092313779 / 20000000000000 : ℝ) * x^0 * y^0 * z^5
  | 9 => 0 + (91975391579759 / 12500000000000 : ℝ) * x^4 * y^1 * z^0 + (-441481879582843 / 10000000000000 : ℝ) * x^2 * y^1 * z^2 + (91975391579759 / 12500000000000 : ℝ) * x^0 * y^1 * z^4
  | _ => 0 + (29085172607791 / 2500000000000 : ℝ) * x^4 * y^0 * z^1 + (-232681380862329 / 10000000000000 : ℝ) * x^2 * y^0 * z^3 + (232681380862329 / 100000000000000 : ℝ) * x^0 * y^0 * z^5

/-- Mathematical Jacobian transpose of those eleven polynomials. All
upstream channels contribute to each of the three coordinate gradients. -/
def jacobianTranspose (x y z : ℝ) (upstream : Fin 11 → ℝ) (component : Fin 3) : ℝ :=
  ∑ channel, upstream channel *
    deriv (fun t => harmonic channel (if component=0 then t else x)
      (if component=1 then t else y) (if component=2 then t else z)) (![x,y,z] component)

/-- The complete forward grid implements the eleven independent polynomials. -/
specification forward (p : Parameters) (fit : p.columnOffset + 11 ≤ p.outputWidth) :
    Spec.Real ((LogicalIO.forwardView p) ⊨ (fun xs row channel =>
      harmonic channel (xs row 0) (xs row 1) (xs row 2))) :=
  (LogicalIO.forward_iff p _).mpr (IOCorrect.real_forward p fit)

/-- Every upstream channel contributes; active initial gradients are prepared zero. -/
specification backward (p : Parameters) (fit : p.columnOffset + 11 ≤ p.outputWidth) :
    Spec.Real ((LogicalIO.backwardView p) ⊨{LogicalIO.Prepared p} (fun xs row component =>
      jacobianTranspose (xs.coordinates row 0) (xs.coordinates row 1) (xs.coordinates row 2)
        (xs.upstream row) component)) :=
  ReaderConnections.backward p fit jacobianTranspose rfl

#specNonCircular harmonic avoiding [forwardKernel, backwardKernel, FP.Structural.exec]
#specNonCircular jacobianTranspose avoiding [forwardKernel, backwardKernel, FP.Structural.exec]
#axiomsClean forward
#axiomsClean backward
#stmtSurfaceSubset forward ⊆
  [LogicalIO.Coordinates, LogicalIO.Harmonics, Parameters.rows, Spec.Real, LogicalProgramIO.Implements, LogicalIO.forwardView, harmonic,
   Parameters, Parameters.columnOffset, Parameters.outputWidth]
#stmtSurfaceSubset backward ⊆
  [LogicalIO.BackwardInputs, LogicalIO.Coordinates, Parameters.rows, Spec.Real, LogicalProgramIO.ImplementsOn, LogicalIO.backwardView, jacobianTranspose,
   LogicalIO.Prepared, Parameters, Parameters.columnOffset, Parameters.outputWidth,
   LogicalIO.BackwardInputs.coordinates, LogicalIO.BackwardInputs.upstream]
end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Correct
