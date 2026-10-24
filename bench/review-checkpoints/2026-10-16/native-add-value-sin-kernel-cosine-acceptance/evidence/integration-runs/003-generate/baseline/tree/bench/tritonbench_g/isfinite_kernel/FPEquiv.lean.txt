import bench.tritonbench_g.isfinite_kernel.Kernels
import bench.tritonbench_g.isfinite_kernel.Proofs.FP
import bench.tritonbench_g.isfinite_kernel.Proofs.WitnessSamples
import VeriTile.Triton.Float.ProfiledRewrite

/-! Ordinary profiled whole-state equivalence of the complete source pair.
The empty assumption list is sufficient because only a natural index product
is commuted. Actual successful complete runs, including both nonempty source
branches over all fourteen special-value samples, accompany that equality. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.Equivalence
open VeriTile VeriTile.Triton FP.Structural FP.ProfiledRewrite
open Parametric TypedBits
open scoped VeriTile.Spec
noncomputable section

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def original32 (p : Parameters) : Program where
  kernel := Kernels.original32 p
  domain := []

def original64 (p : Parameters) : Program where
  kernel := Kernels.original64 p
  domain := []

def indexCommuted32 (p : Parameters) : Program where
  kernel := Kernels.indexCommuted32 p
  domain := []

def indexCommuted64 (p : Parameters) : Program where
  kernel := Kernels.indexCommuted64 p
  domain := []

specification isfinite_fp32_equiv (p : Parameters) :
    original32 p ≡[assumptions] indexCommuted32 p := by
  apply Spec.FloatingPoint.ofNumerical
    (lhs := original32 p) (rhs := indexCommuted32 p)
    (structural := fun _ _ => False) rfl rfl
  intro α _ M D _ initial _
  exact whole32 p (engine M) initial

specification isfinite_fp64_equiv (p : Parameters) :
    original64 p ≡[assumptions] indexCommuted64 p := by
  apply Spec.FloatingPoint.ofNumerical
    (lhs := original64 p) (rhs := indexCommuted64 p)
    (structural := fun _ _ => False) rfl rfl
  intro α _ M D _ initial _
  exact whole64 p (engine M) initial

theorem successful32_pair (rest : Algebra Float32Bits) (p : Parameters)
    (initial : State Float32Bits) :
    ∃ final, FP.Structural.exec (engine (backend32 rest)) (original32 p).kernel initial = some final ∧
      FP.Structural.exec (engine (backend32 rest)) (indexCommuted32 p).kernel initial = some final := by
  obtain ⟨final, ho, hc, _⟩ := pair32_run rest p initial
  exact ⟨final, ho, hc⟩

theorem successful64_pair (rest : Algebra Float64Bits) (p : Parameters)
    (initial : State Float64Bits) :
    ∃ final, FP.Structural.exec (engine (backend64 rest)) (original64 p).kernel initial = some final ∧
      FP.Structural.exec (engine (backend64 rest)) (indexCommuted64 p).kernel initial = some final := by
  obtain ⟨final, ho, hc, _⟩ := pair64_run rest p initial
  exact ⟨final, ho, hc⟩

def witnessParameters (oneTile : Bool) : Parameters where
  input := "input"
  output := "output"
  inputStride := 1
  inputOrder := 0
  outputStride := 2
  outputOrder := 0
  size := 14
  numTasks := 14
  iterations := 2
  block := if oneTile then 16 else 8
  oneTile := oneTile
  inputOrderZero := rfl
  outputOrderZero := rfl

/-- Both full source branches classify both signs of zero, subnormal, one,
max finite, infinity, quiet NaN and signaling NaN. All seventeen sentinel
cells remain unchanged, and omitted padding returns signaling NaN. -/
theorem nonempty32_pair (rest : Algebra Float32Bits) (oneTile : Bool) :
    ∃ final, FP.Structural.exec (engine (backend32 rest))
        (original32 (witnessParameters oneTile)).kernel initial32 = some final ∧
      FP.Structural.exec (engine (backend32 rest))
        (indexCommuted32 (witnessParameters oneTile)).kernel initial32 = some final ∧
      observation final = expectedObservation := by
  obtain ⟨final, ho, hv⟩ := successful32_both_branches rest oneTile
  exact ⟨final, ho, (whole32 (witnessParameters oneTile) _ initial32).symm.trans ho, hv⟩

theorem nonempty64_pair (rest : Algebra Float64Bits) (oneTile : Bool) :
    ∃ final, FP.Structural.exec (engine (backend64 rest))
        (original64 (witnessParameters oneTile)).kernel initial64 = some final ∧
      FP.Structural.exec (engine (backend64 rest))
        (indexCommuted64 (witnessParameters oneTile)).kernel initial64 = some final ∧
      observation final = expectedObservation := by
  obtain ⟨final, ho, hv⟩ := successful64_both_branches rest oneTile
  exact ⟨final, ho, (whole64 (witnessParameters oneTile) _ initial64).symm.trans ho, hv⟩

#axiomsClean isfinite_fp32_equiv
#axiomsClean isfinite_fp64_equiv
#axiomsClean successful32_pair
#axiomsClean successful64_pair
#axiomsClean nonempty32_pair
#axiomsClean nonempty64_pair
end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.Equivalence
