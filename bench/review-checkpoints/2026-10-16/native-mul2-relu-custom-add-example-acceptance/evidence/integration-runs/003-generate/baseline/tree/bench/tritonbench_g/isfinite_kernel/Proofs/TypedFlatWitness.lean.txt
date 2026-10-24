import bench.tritonbench_g.isfinite_kernel.Proofs.TypedFlatCorrect

/-! Concrete finite allocations satisfy the complete paired flat contracts.
Both source branches run on fourteen special bit patterns with nonzero base
addresses, output stride two, masked tails and signaling-NaN padding. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open Parametric TypedBits BitClassification
noncomputable section

def flatWitnessAlloc : FlatAlloc where
  flat := "flat"
  regions := ["input", "output"]
  base := fun r => if r = "input" then 32 else if r = "output" then 64 else 0
  extent := fun r => if r = "input" then 16 else if r = "output" then 32 else 0

def flatWitnessParameters (oneTile : Bool) : Parameters where
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

def flatWitnessBackground {α : Type} : RegionName → Nat → Cell α :=
  fun _ _ => .mk .nat 1729

theorem flatWitnessAlloc_disjoint : flatWitnessAlloc.Disjoint := by
  intro r hr r' hr' hne
  simp only [flatWitnessAlloc, List.mem_cons, List.not_mem_nil, or_false] at hr hr'
  rcases hr with rfl | rfl <;> rcases hr' with rfl | rfl <;>
    simp_all [flatWitnessAlloc]

theorem nonempty32_flat_pair (rest : Algebra Float32Bits) (oneTile : Bool) :
    ∃ final,
      FP.Structural.exec (engine (backend32 rest))
        (flattenComputeKernel flatWitnessAlloc (original32 (flatWitnessParameters oneTile)))
        (flattenState flatWitnessAlloc flatWitnessBackground initial32.undef initial32) = some final ∧
      FP.Structural.exec (engine (backend32 rest))
        (flattenComputeKernel flatWitnessAlloc (candidate32 (flatWitnessParameters oneTile)))
        (flattenState flatWitnessAlloc flatWitnessBackground initial32.undef initial32) = some final ∧
      FlatPost flatWitnessAlloc finite32 (flatWitnessParameters oneTile) sample32
        (flattenState flatWitnessAlloc flatWitnessBackground initial32.undef initial32) final := by
  apply pair32_flat_correct flatWitnessAlloc flatWitnessAlloc_disjoint flatWitnessBackground
    initial32.undef rest (flatWitnessParameters oneTile) sample32 initial32
  · intro r o
    rfl
  · simp [flatWitnessAlloc, flatWitnessParameters]
  · simp [flatWitnessAlloc, flatWitnessParameters]
  · intro x hx
    simp only [flatWitnessParameters] at hx
    simp [flatWitnessParameters, flatWitnessAlloc]
    omega
  · intro x hx
    simp only [flatWitnessParameters] at hx
    simp [flatWitnessParameters, flatWitnessAlloc]
    omega
  · simp [flatWitnessParameters]
  · simp [flatWitnessParameters]
  · intro x hx
    simp only [flatWitnessParameters] at hx
    simp [flatWitnessParameters, initial32, hx]

theorem nonempty64_flat_pair (rest : Algebra Float64Bits) (oneTile : Bool) :
    ∃ final,
      FP.Structural.exec (engine (backend64 rest))
        (flattenComputeKernel flatWitnessAlloc (original64 (flatWitnessParameters oneTile)))
        (flattenState flatWitnessAlloc flatWitnessBackground initial64.undef initial64) = some final ∧
      FP.Structural.exec (engine (backend64 rest))
        (flattenComputeKernel flatWitnessAlloc (candidate64 (flatWitnessParameters oneTile)))
        (flattenState flatWitnessAlloc flatWitnessBackground initial64.undef initial64) = some final ∧
      FlatPost flatWitnessAlloc finite64 (flatWitnessParameters oneTile) sample64
        (flattenState flatWitnessAlloc flatWitnessBackground initial64.undef initial64) final := by
  apply pair64_flat_correct flatWitnessAlloc flatWitnessAlloc_disjoint flatWitnessBackground
    initial64.undef rest (flatWitnessParameters oneTile) sample64 initial64
  · intro r o
    rfl
  · simp [flatWitnessAlloc, flatWitnessParameters]
  · simp [flatWitnessAlloc, flatWitnessParameters]
  · intro x hx
    simp only [flatWitnessParameters] at hx
    simp [flatWitnessParameters, flatWitnessAlloc]
    omega
  · intro x hx
    simp only [flatWitnessParameters] at hx
    simp [flatWitnessParameters, flatWitnessAlloc]
    omega
  · simp [flatWitnessParameters]
  · simp [flatWitnessParameters]
  · intro x hx
    simp only [flatWitnessParameters] at hx
    simp [flatWitnessParameters, initial64, hx]

#axiomsClean flatWitnessAlloc_disjoint
#axiomsClean nonempty32_flat_pair
#axiomsClean nonempty64_flat_pair
end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
