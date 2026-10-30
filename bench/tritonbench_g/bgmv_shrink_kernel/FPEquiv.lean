import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.PairIO

/-! Complete original/scalar-left program equivalence. The sole numerical
rewrite reuses the strict all-word FP32 multiplication certificate. The full
FP16/BF16 sources, output conversion, signed metadata, batch × 64 launch and
each address's independent atomic order remain explicit. -/
namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.Equivalence
open VeriTile Triton FP.Structural FPValues
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset
open scoped VeriTile.Spec
noncomputable section

abbrev original {A : Type} (M : Algebra A) (P : Primitives A) (h : Call)
    (indices : Nat → Int) (scale : A) (orders : Orders h) :=
  PerAddressProgramIO.Implementation.original
    (IO.programIO M P h indices scale orders id) (IO.domain h indices)

abbrev scalarLeft {A : Type} (M : Algebra A) (P : Primitives A) (h : Call)
    (indices : Nat → Int) (scale : A) (orders : Orders h) :=
  PerAddressProgramIO.Implementation.ofVariant
    (PairIO.candidate M P h indices scale orders) (IO.domain h indices)

/-- No finite-value restriction and no experimental rule is required.
Both actual kernels execute safely and successfully on the full retained
cell domain, with equal tagged outputs and both exterior memory frames. -/
specification scalar_left_strict (M : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (P : Primitives (FP.Value .fp32)) (h : Call) (indices : Nat → Int)
    (scale : FP.Value .fp32) (orders : Orders h) :
    original (PairIO.strictAlgebra M cfg) P h indices scale orders ≡[[]]
      scalarLeft (PairIO.strictAlgebra M cfg) P h indices scale orders := by
  exact PerAddressProgramIO.Implementation.of_equivalentOnCells
    (PairIO.candidate _ P h indices scale orders)
    (PairIO.equivalent _ P h indices scale orders (PairIO.strict_multiplication M cfg))

#axiomsClean scalar_left_strict
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.Equivalence
