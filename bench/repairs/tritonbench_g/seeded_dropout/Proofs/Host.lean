import bench.repairs.tritonbench_g.seeded_dropout.Proofs.Contract
import bench.repairs.tritonbench_g.seeded_dropout.Proofs.Source
import VeriTile.Triton.Float.ProfiledRewrite

namespace VeriTile.Bench.Repairs.SeededDropout
open VeriTile.Triton VeriTile.Torch
open TritonBenchG.SeededDropout.Whole
noncomputable section

/-- Ordinary agreement between API metadata and the Tensor passed to the source.
The no-negative guard makes direct storage reads logical FP32 reads. -/
structure MetadataBinding (x : TensorMetadata) (input : Tensor) : Prop where
  shape : x.shape = input.shape
  contiguous : x.isContiguous = input.contiguous
  dtype : x.dtype = .floating .fp32 → input.dtype = .fp32
  device : x.device.index = input.cudaDevice

/-- Ordinary live allocation facts, supplied by Tensor and empty_like APIs.
There is no dtype, size, scalar, or padded-pointer restriction hidden here. -/
structure AllocationAPI (a : Allocated α input entry preexisting) where
  capacity : Nat → Nat
  inputBase : input.base ≤ capacity input.allocation
  outputBase : a.out.base ≤ capacity a.out.allocation
  inputBounds : ∀ i, i < input.numel → input.base+i < capacity input.allocation
  outputBounds : ∀ i, i < input.numel → a.out.base+i < capacity a.out.allocation

def admittedDomain (x : TensorMetadata) (p : ℝ) (seed : Int) (h : InputContract x p seed)
    (binding : MetadataBinding x input) (a : Allocated α input entry preexisting)
    (api : AllocationAPI a) : Domain a where
  dtype := binding.dtype h.tensor.dtype.1
  size := by simpa [VeriTile.Torch.TensorMetadata.numel,Tensor.numel,binding.shape] using h.tensor.extent
  capacity := api.capacity
  inputBase := api.inputBase
  outputBase := api.outputBase
  inputBounds := api.inputBounds
  outputBounds := api.outputBounds

def selectedExecute [Inhabited α] (M : FP.Structural.Algebra α) (p : ℝ) (s : Int)
    (rt : Runtime α input entry preexisting) : Outcome α :=
  let selected := selectSeed s
  executeHost M (legacyKind selected.1) p (legacySeed selected.1 selected.2) rt

/-- Checked admission precedes the retained original allocation/assert/launch body. -/
def runChecked [Inhabited α] (M : FP.Structural.Algebra α) (x : TensorMetadata) (p : ℝ) (s : Int)
    (rt : Runtime α input entry preexisting) : Except InputError (Outcome α) := do
  checkInputs x (.finite p) (.integer s)
  pure (selectedExecute M p s rt)

theorem checked_eq_source [Inhabited α] (M : FP.Structural.Algebra α)
    (x : TensorMetadata) (p : ℝ) (s : Int) (h : InputContract x p s)
    (rt : Runtime α input entry preexisting) :
    runChecked M x p s rt = .ok (selectedExecute M p s rt) := by
  have ok := (checkInputs_iff x (.finite p) (.integer s)).mpr h
  simp [runChecked,ok]

theorem checked_real (x : TensorMetadata) (p : ℝ) (s : Int) (h : InputContract x p s)
    (binding : MetadataBinding x input) (rt : Runtime ℝ input entry preexisting)
    (api : ∀ a, rt.allocate = .ok a → AllocationAPI a) :
    ∃ outcome, runChecked realAlgebra x p s rt = .ok outcome ∧
      HostSpec (realFormula p s) rt outcome := by
  refine ⟨selectedExecute realAlgebra p s rt,checked_eq_source _ x p s h rt,?_⟩
  obtain ⟨abi,value⟩ := selected_seed s h.seedRange
  have result := real_host_outcomes (legacyKind (selectSeed s).1) p
    (legacySeed (selectSeed s).1 (selectSeed s).2)
    (PythonSeedABI.valid _ _ (legacy_seed_abi _ _ abi)) rt
    (fun a ha _ => admittedDomain x p s h binding a (api a ha))
  simpa only [selectedExecute,legacy_seed_value,value] using result

theorem checked_fp32 (M : FP.Structural.Algebra (BitVec 32))
    (lt : BitVec 32 → BitVec 32 → Bool)
    (compare : M.compareLt (some .fp32) .real = some lt)
    (x : TensorMetadata) (p : ℝ) (s : Int) (h : InputContract x p s)
    (binding : MetadataBinding x input) (rt : Runtime (BitVec 32) input entry preexisting)
    (api : ∀ a, rt.allocate = .ok a → AllocationAPI a) :
    ∃ outcome, runChecked (FP.ProfiledRewrite.engine M) x p s rt = .ok outcome ∧
      HostSpec (scalar (FP.ProfiledRewrite.engine M) lt p s) rt outcome := by
  refine ⟨selectedExecute (FP.ProfiledRewrite.engine M) p s rt,checked_eq_source _ x p s h rt,?_⟩
  obtain ⟨abi,value⟩ := selected_seed s h.seedRange
  have result := host_outcomes (FP.ProfiledRewrite.engine M) lt compare (legacyKind (selectSeed s).1) p
    (legacySeed (selectSeed s).1 (selectSeed s).2)
    (PythonSeedABI.valid _ _ (legacy_seed_abi _ _ abi)) rt
    (fun a ha _ => admittedDomain x p s h binding a (api a ha))
  simpa only [selectedExecute,legacy_seed_value,value] using result

#axiomsClean checked_real
#axiomsClean checked_fp32
end
end VeriTile.Bench.Repairs.SeededDropout
