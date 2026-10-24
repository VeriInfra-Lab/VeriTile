import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Reference
import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Polynomial

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.ScalarReal
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 64000
set_option linter.unusedSimpArgs false

/-- Independent Real interpretation of the actually used scalar primitives.
It does not claim that a floating computation decodes to its exact real value. -/
def backend (rest : Algebra ℝ) : Algebra ℝ :=
  {rest with
    fp32Load:=id
    literal:=fun _ _ x=>x
    cast:=fun _ _ _ x=>x
    binary:=fun p d op x y=>match op with
      | .add=>x+y
      | .sub=>x-y
      | .mul=>x*y
      | _=>rest.binary p d op x y
    unary:=fun p op x=>match op with
      | .neg _=>-x
      | _=>rest.unary p op x}

theorem forward (rest : Algebra ℝ) (x y z : ℝ) (channel : Fin 11) :
    Reference.forwardValues (backend rest) x y z channel=Polynomial.value channel x y z := by
  fin_cases channel <;>
    norm_num [Reference.forwardValues,
      VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP.channels,
      backend,Polynomial.value] <;> ring

/-- All three repaired backward expressions are the true Jacobian-transpose
product of the independently stated decimal polynomials. The initial gradient
is the actual fresh zeros allocation, before the source's retained += step. -/
theorem backward (rest : Algebra ℝ) (x y z : ℝ) (upstream : Fin 11 → ℝ)
    (component : Fin 3) :
    Reference.backwardValues (backend rest) x y z upstream (fun _=>0) component=
      Polynomial.vjp x y z upstream component := by
  have expanded : upstream=![upstream 0,upstream 1,upstream 2,upstream 3,upstream 4,
      upstream 5,upstream 6,upstream 7,upstream 8,upstream 9,upstream 10] := by
    funext c
    fin_cases c <;> rfl
  rw [expanded]
  fin_cases component <;>
    norm_num [Reference.backwardValues,backend,Polynomial.vjp,
      Polynomial.jacobian,Fin.sum_univ_succ,Fin.succ] <;> ring!

#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.ScalarReal
