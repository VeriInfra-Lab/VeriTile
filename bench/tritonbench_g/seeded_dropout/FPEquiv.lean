import bench.tritonbench_g.seeded_dropout.Proofs.FP
import VeriTile.Triton.Float.ProfiledRewrite

/-! Typed implementation equivalence for the complete original source and
integer-index-commuted candidate. All floating operations, canonical random
bits, comparison, eager divide and source storage conversion are shared.
The theorem preserves full states and failure from arbitrary initial states.
No numerical atom or operand-domain restriction is assumed. -/
namespace VeriTile.Bench.TritonBenchG.SeededDropout.Equivalence
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def original (storage : Storage) (kind : Random.IntegerType) (x y : RegionName)
    (N : Nat) (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (hB : SourceDomain.LegalBlock B) : Program where
  kernel := source storage kind Bool.false x y N p seed B hB
  domain := []

def indexCommuted (storage : Storage) (kind : Random.IntegerType) (x y : RegionName)
    (N : Nat) (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (hB : SourceDomain.LegalBlock B) : Program where
  kernel := source storage kind Bool.true x y N p seed B hB
  domain := []

specification seeded_dropout_equiv (storage : Storage) (kind : Random.IntegerType)
    (x y : RegionName) (N : Nat) (p : ℝ) (seed : SeedCarrier kind)
    (B : Nat) (hB : SourceDomain.LegalBlock B) :
    original storage kind x y N p seed B hB ≡[assumptions]
      indexCommuted storage kind x y N p seed B hB := by
  apply Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl
    (by cases storage <;> cases kind <;> rfl)
  intro α _ M D _ u _
  exact FPProof.whole_state storage kind x y N p seed B hB (engine M) u

end VeriTile.Bench.TritonBenchG.SeededDropout.Equivalence
