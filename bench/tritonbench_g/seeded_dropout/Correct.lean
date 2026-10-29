import bench.tritonbench_g.seeded_dropout.Proofs.WholePublic
import bench.tritonbench_g.seeded_dropout.Proofs.RealSurface
import VeriTile.Meta.Specification

/-! Complete seeded-dropout source correctness on the standard region
realization/frame surface and successful finite flat memory execution.
No existing KernelIO skin packages the runtime seed ABI and storage profiles.
The fixed canonical Philox generator includes its actual fp32 conversion and
scaling roundings; the surrounding real formula erases source precision.
Original runtime seed/p and all source statements remain. Runtime seed ABI,
legal block size, launch/index domain and finite active-access bounds are
explicit. Per-program input/output aliases are allowed. The current whole-state
flatten bridge requires zero undef explicitly; the source load still omits
padding. Host execution, concurrent aliased-program race freedom and physical
backend rounding correctness are not asserted by this real specification. -/
namespace VeriTile.Bench.TritonBenchG.SeededDropout.Correctness
open VeriTile VeriTile.Triton TypedSource
noncomputable section

def original (storage : Storage) (kind : Random.IntegerType) (x y : RegionName)
    (N : Nat) (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (hB : SourceDomain.LegalBlock B) :
    ComputeKernel := (source storage kind Bool.false x y N p seed B hB).eraseDType

def indexCommuted (storage : Storage) (kind : Random.IntegerType) (x y : RegionName)
    (N : Nat) (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (hB : SourceDomain.LegalBlock B) :
    ComputeKernel := (source storage kind Bool.true x y N p seed B hB).eraseDType

specification seeded_dropout_correctness (storage : Storage) (kind : Random.IntegerType)
    (x y : RegionName) (N : Nat) (p : ℝ) (seed : SeedCarrier kind)
    (B : Nat) (hB : SourceDomain.LegalBlock B) :
    Spec.Real (
      (∀ u : BlockState, N < 2^31 → u.pid < SourceDomain.grid N B →
        kind.inRange (seedInt kind seed) = Bool.true →
        ComputeCorrect.RealizesFrame_without_Rounding (α := MemCell)
          (original storage kind x y N p seed B hB) u
          (RealSurface.writes y N B u.pid) (RealSurface.expected x p (seedInt kind seed) B u)) ∧
      RealIO.Implements (original storage kind x y N p seed B hB).toAlgKernel
        kind x y N p seed B) := by
  constructor
  · exact RealSurface.realizes_frame storage kind Bool.false x y N p seed B hB
  · exact RealIO.original_correct storage kind x y N p seed B hB

specification index_commuted_correctness (storage : Storage) (kind : Random.IntegerType)
    (x y : RegionName) (N : Nat) (p : ℝ) (seed : SeedCarrier kind)
    (B : Nat) (hB : SourceDomain.LegalBlock B) :
    Spec.Real (
      (∀ u : BlockState, N < 2^31 → u.pid < SourceDomain.grid N B →
        kind.inRange (seedInt kind seed) = Bool.true →
        ComputeCorrect.RealizesFrame_without_Rounding (α := MemCell)
          (indexCommuted storage kind x y N p seed B hB) u
          (RealSurface.writes y N B u.pid) (RealSurface.expected x p (seedInt kind seed) B u)) ∧
      RealIO.Implements (indexCommuted storage kind x y N p seed B hB).toAlgKernel
        kind x y N p seed B) := by
  constructor
  · exact RealSurface.realizes_frame storage kind Bool.true x y N p seed B hB
  · exact RealIO.candidate_correct storage kind x y N p seed B hB

end
end VeriTile.Bench.TritonBenchG.SeededDropout.Correctness

namespace VeriTile.Bench.TritonBenchG.SeededDropout.Whole
open VeriTile VeriTile.Triton TypedSource
noncomputable section

/-- Full original Python entry with an independent mathematical dropout
formula, all original branches, complete frame and finite-index correspondence. -/
specification seeded_dropout_real_source_outcomes (kind : Random.IntegerType) (p : ℝ)
    (seed : SeedCarrier kind) (_probability : ProbabilityDomain p) (abi : PythonSeedABI kind seed)
    (rt : Runtime ℝ input entry preexisting)
    (valid : ∀ a, rt.allocate = .ok a → input.contiguous = Bool.true → Domain a)
    (finite : ∀ a (ha : rt.allocate = .ok a) (hc : input.contiguous = Bool.true),
      FiniteAdmission a (valid a ha hc)) :
    Spec.Real (HostSpec (realFormula p (seedInt kind seed)) rt (executeHost realAlgebra kind p seed rt) ∧
      ∀ a ha hc, IndexCorrespondence a (valid a ha hc) (finite a ha hc)) := by
  exact ⟨real_host_outcomes kind p seed (PythonSeedABI.valid kind seed abi) rt valid,
    fun a ha hc => index_correspondence a (valid a ha hc) (finite a ha hc)⟩

#axiomsClean seeded_dropout_real_source_outcomes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.SeededDropout.Whole
