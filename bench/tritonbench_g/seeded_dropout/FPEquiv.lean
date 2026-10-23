import bench.tritonbench_g.seeded_dropout.Proofs.WholePublic
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

namespace VeriTile.Bench.TritonBenchG.SeededDropout.Whole
open VeriTile.Triton TypedSource FP.Structural
noncomputable section

/-- All 32-bit payloads remain admitted, including infinities and NaNs.
The exact primitive implementations are explicit; no finite-only or whole-result
law is assumed. The default engine resolves the source comparison's None tag. -/
specification seeded_dropout_fp32_source_outcomes (M : Algebra (BitVec 32))
    (lt : BitVec 32 → BitVec 32 → Bool)
    (compare : M.compareLt (some .fp32) .real = some lt)
    (kind : Random.IntegerType) (p : ℝ) (seed : SeedCarrier kind)
    (_probability : ProbabilityDomain p) (abi : PythonSeedABI kind seed)
    (rt : Runtime (BitVec 32) input entry preexisting)
    (valid : ∀ a, rt.allocate = .ok a → input.contiguous = Bool.true → Domain a)
    (finite : ∀ a (ha : rt.allocate = .ok a) (hc : input.contiguous = Bool.true),
      FiniteAdmission a (valid a ha hc)) :
    HostSpec (scalar (FP.ProfiledRewrite.engine M) lt p (seedInt kind seed)) rt
      (executeHost (FP.ProfiledRewrite.engine M) kind p seed rt) ∧
      ∀ a ha hc, IndexCorrespondence a (valid a ha hc) (finite a ha hc) := by
  exact ⟨host_outcomes (FP.ProfiledRewrite.engine M) lt compare kind p seed
    (PythonSeedABI.valid kind seed abi) rt valid,
    fun a ha hc => index_correspondence a (valid a ha hc) (finite a ha hc)⟩

#axiomsClean seeded_dropout_fp32_source_outcomes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.SeededDropout.Whole
