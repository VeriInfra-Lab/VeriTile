import bench.repairs.tritonbench_g.seeded_dropout.Proofs.KernelExecution
import VeriTile.Triton.Float.ProfiledRewrite

namespace VeriTile.Bench.Repairs.SeededDropout.KernelEquivalence
open VeriTile.Triton FP.Structural KernelExecution
open scoped VeriTile.Spec
noncomputable section

def assumptions : Spec.Assumptions FP.GuardedFragment := []
def original (k : SeedKind) (X O : RegionName) (n : Nat) (p : ℝ) (seed : SeedCarrier k) :
    FP.ProfiledRewrite.Program where
  kernel := kernel k X O n p seed
  domain := []
def indexCommuted (k : SeedKind) (X O : RegionName) (n : Nat) (p : ℝ) (seed : SeedCarrier k) :
    FP.ProfiledRewrite.Program where
  kernel := indexCommutedKernel k X O n p seed
  domain := []

/-- Unconditional source-state equality, including failure, from the existing
integer multiplication law. Successful execution is proved separately below. -/
theorem index_commuted_equiv (k : SeedKind) (X O : RegionName) (n : Nat)
    (p : ℝ) (seed : SeedCarrier k) :
    original k X O n p seed ≡[assumptions] indexCommuted k X O n p seed := by
  apply VeriTile.Spec.FloatingPoint.ofNumerical
  · rfl
  · cases k <;> rfl
  intro α _ M D _ s _
  exact execution_eq (FP.ProfiledRewrite.engine M) k X O n p seed s

/-- Both actual implementations succeed and agree on the entire typed memory.
The seed/index domain and available comparison primitive are structural model
conditions. No input-value, denominator, RNG or rounding rewrite law is assumed. -/
theorem successful_index_commutation {α : Type} [Inhabited α]
    (M : Algebra α) (lt : α → α → Bool)
    (comparison : M.compareLt (some .fp32) .real = some lt)
    (k : SeedKind) (X O : RegionName) (n : Nat) (p : ℝ) (seed : SeedCarrier k)
    (pid : Nat) (s : State α) (hn : n < 2^31) (hp : pid < gridCount n) (hs : SeedABI k seed) :
    ∃ a b,
      FP.Structural.exec (FP.ProfiledRewrite.engine M) (original k X O n p seed).kernel (entry pid s) = some a ∧
      FP.Structural.exec (FP.ProfiledRewrite.engine M) (indexCommuted k X O n p seed).kernel (entry pid s) = some b ∧
      a.mem = b.mem ∧
      (∀ r i, Outside O n pid r i → a.mem r i = s.mem r i) := by
  let result := originalExecution (FP.ProfiledRewrite.engine M) lt comparison k X O n p seed pid s hn hp hs
  refine ⟨result.val,result.val,result.property,?_,rfl,?_⟩
  · exact (execution_eq (FP.ProfiledRewrite.engine M) k X O n p seed (entry pid s)).symm.trans result.property
  · exact frame (FP.ProfiledRewrite.engine M) lt comparison k X O n p seed pid s hn hp hs

theorem original_source (k : SeedKind) (n : Nat) (p : ℝ) (seed : SeedCarrier k) :
    (original k "x_ptr" "output_ptr" n p seed).kernel = originalKernel k n p seed := rfl

theorem candidate_source (k : SeedKind) (n : Nat) (p : ℝ) (seed : SeedCarrier k) :
    (indexCommuted k "x_ptr" "output_ptr" n p seed).kernel = candidateKernel k n p seed := rfl

#print_fp_assumptions index_commuted_equiv
#axiomsClean index_commuted_equiv
#axiomsClean successful_index_commutation
#axiomsClean original_source
#axiomsClean candidate_source
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SeededDropout.KernelEquivalence
