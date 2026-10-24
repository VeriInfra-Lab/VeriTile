import bench.tritonbench_g.uniform_sampling.Proofs.FP
import bench.tritonbench_g.uniform_sampling.Proofs.TypedExecution
import VeriTile.Triton.Float.ProfiledRewrite

/-! Complete typed source/candidate equivalence with no floating rewrite
assumptions. The only edit commutes an integer multiplication. All generator
fields, uniform bit payloads, floating operations and stores remain shared.
This equality includes failure; typed success is a separate proved property.
It does not identify the unfused source evaluator with native FMA lowering.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.Equivalence
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def original (storage : Storage) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (hB : SourceDomain.LegalBlock B) : Program where
  kernel := source storage Bool.false y N seed offset lower upper B hB
  domain := []

def indexCommuted (storage : Storage) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (hB : SourceDomain.LegalBlock B) : Program where
  kernel := source storage Bool.true y N seed offset lower upper B hB
  domain := []

specification uniform_sampling_equiv (storage : Storage) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : ℝ) (B : Nat)
    (hB : SourceDomain.LegalBlock B) :
    original storage y N seed offset lower upper B hB ≡[assumptions]
      indexCommuted storage y N seed offset lower upper B hB := by
  apply Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl
    (by cases storage <;> rfl)
  intro α _ M D _ u _
  exact FPProof.whole_state storage y N seed offset lower upper B hB (engine M) u

#axiomsClean uniform_sampling_equiv

/-- The public programs share a successful state in the same FP engine,
from empty registers and the original nonempty wrapper configuration. -/
theorem nonempty_pair (storage : Storage) :
    ∃ v,
      FP.Structural.exec (engine TypedExecution.consistencyModel)
        (original storage "out" 4095 (-1) (-1) 0 1 1024 TypedExecution.block_legal).kernel
        TypedExecution.initial = some v ∧
      FP.Structural.exec (engine TypedExecution.consistencyModel)
        (indexCommuted storage "out" 4095 (-1) (-1) 0 1 1024 TypedExecution.block_legal).kernel
        TypedExecution.initial = some v :=
  TypedExecution.nonempty_pair storage

#axiomsClean nonempty_pair

end VeriTile.Bench.TritonBenchG.UniformSampling.Equivalence
