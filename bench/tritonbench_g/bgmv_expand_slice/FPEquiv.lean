import bench.tritonbench_g.bgmv_expand_slice.Proofs.PairIO

/-! Complete signed-source output-first equivalence. The shared strict FP32
software addition certificate covers all words. Other local operations and the
FP16 reduction plan are fixed identically in both actual programs. The theorem
makes no hardware-conformance claim. -/
namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Equivalence
open VeriTile Triton FP.Structural Direct Direct.FPValues
open scoped VeriTile.Spec
noncomputable section

abbrev original {A : Type} (M : Algebra A) (R : Reduction A) (h : Call) :=
  ProgramIO.Implementation.original (PublicIO.sourceIO M R id h) PublicIO.Valid
abbrev outputFirst {A : Type} (M : Algebra A) (R : Reduction A) (h : Call) :=
  ProgramIO.Implementation.ofVariant (PairIO.candidate M R h) PublicIO.Valid

/-- Strict all-word instance of the primary pair. This has no finite-value
restriction and no experimental numerical rule. -/
specification output_first_strict (M : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (R : Reduction (FP.Value .fp32)) (h : Call) :
    original (PairIO.strictAlgebra M cfg) R h ≡[[]]
      outputFirst (PairIO.strictAlgebra M cfg) R h := by
  exact ProgramIO.Implementation.of_equivalentOn (PairIO.candidate _ R h)
    (PairIO.equivalent _ R h (PairIO.strict_addition M cfg))

#axiomsClean output_first_strict
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Equivalence
