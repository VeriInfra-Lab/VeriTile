import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Polynomial

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.HeadlinePolynomial
noncomputable section

/-- Reuse the existing Mathlib derivative certificate for the independent
polynomial table; this is a mathematical bridge, with no kernel assumption. -/
theorem jacobian_eq_deriv (channel : Fin 11) (component : Fin 3) (x y z : ℝ) :
    Polynomial.jacobian channel component x y z =
      deriv (fun t => Polynomial.value channel (if component=0 then t else x)
        (if component=1 then t else y) (if component=2 then t else z))
        (Polynomial.coordinate component x y z) :=
  (Polynomial.exact_partial_derivative channel component x y z).deriv.symm

#axiomsClean jacobian_eq_deriv
end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.HeadlinePolynomial
