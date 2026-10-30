import bench.tritonbench_g.flash_decode2_phi.Proofs.FPSoftware
import bench.tritonbench_g.flash_decode2_phi.Proofs.FPShape

/-! Previously accepted supplied-state Phi stage contracts. The arithmetic
faces expose output cells and all untouched memory. The separate maximum
recurrence remains a recurrence identity, not execution evidence. Exp32 and
native/compiler bindings remain explicit outstanding obligations. -/
namespace VeriTile.Bench.TritonBenchG.FlashDecode2Phi.FPCorrect
open VeriTile.Triton FP.Structural FPStages
open FlashDecode2Llama.FPStages (Word address scalarAddress ScalarPost scalar_post)
noncomputable section

specification normalization_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (Acc SumExp Out : RegionName) (p : OutputParams) (s : State α)
    (hinj : Function.Injective (address s p.output (D := p.width))) :
    ∃ t, FP.Structural.exec M (normalization Acc SumExp Out p) s = some t ∧
      Post s t Out p.output p.headDim p.width (normalizationValues M s Acc SumExp p) := by
  obtain ⟨t, he, hm⟩ := normalization_memory M Acc SumExp Out p s
  exact ⟨t, he, post_of_memory s t Out p.output p.headDim p.width _ hinj hm⟩

specification copy_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (Final Out : RegionName) (p : OutputParams) (s : State α)
    (hinj : Function.Injective (address s p.output (D := p.width))) :
    ∃ t, FP.Structural.exec M (copy Final Out p) s = some t ∧
      Post s t Out p.output p.headDim p.width (copyValues M s Final p) := by
  obtain ⟨t, he, hm⟩ := copy_memory M Final Out p s
  exact ⟨t, he, post_of_memory s t Out p.output p.headDim p.width _ hinj hm⟩

specification accumulator_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (Mid Log AccIn Prev Next AccOut : RegionName) (p : StepParams) (s : State α)
    (hinj : Function.Injective (address s p.acc (D := p.width))) :
    ∃ t, FP.Structural.exec M (accumulator Mid Log AccIn Prev Next AccOut p) s = some t ∧
      Post s t AccOut p.acc p.headDim p.width (accumulatorValues M s Mid Log AccIn Prev Next p) := by
  obtain ⟨t, he, hm⟩ := accumulator_memory M Mid Log AccIn Prev Next AccOut p s
  exact ⟨t, he, post_of_memory s t AccOut p.acc p.headDim p.width _ hinj hm⟩

specification sum_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (Log SumIn Prev Next SumOut : RegionName) (p : StepParams) (s : State α) :
    ∃ t, FP.Structural.exec M (sumStep Log SumIn Prev Next SumOut p) s = some t ∧
      ScalarPost s t SumOut p.sum (sumValue M s Log SumIn Prev Next p) := by
  obtain ⟨t, he, hm⟩ := sum_memory M Log SumIn Prev Next SumOut p s
  exact ⟨t, he, scalar_post s t SumOut p.sum _ hm⟩

specification normalization_software_fp32 [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (exp32 : Word → Word) (Acc SumExp Out : RegionName) (p : OutputParams) (s : State Word)
    (hinj : Function.Injective (address s p.output (D := p.width))) :
    ∃ t, FP.Structural.exec (bitBackend rest cfg exp32) (normalization Acc SumExp Out p) s = some t ∧
      Post s t Out p.output p.headDim p.width (roundedNormalization cfg s Acc SumExp p) := by
  simpa only [normalization_software] using
    normalization_fp32_stage (bitBackend rest cfg exp32) Acc SumExp Out p s hinj

specification copy_software_fp32 [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (exp32 : Word → Word) (Final Out : RegionName) (p : OutputParams) (s : State Word)
    (hinj : Function.Injective (address s p.output (D := p.width))) :
    ∃ t, FP.Structural.exec (bitBackend rest cfg exp32) (copy Final Out p) s = some t ∧
      Post s t Out p.output p.headDim p.width (copiedBits s Final p) := by
  simpa only [copy_software] using
    copy_fp32_stage (bitBackend rest cfg exp32) Final Out p s hinj

specification accumulator_software_fp32 [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (exp32 : Word → Word) (Mid Log AccIn Prev Next AccOut : RegionName) (p : StepParams) (s : State Word)
    (hinj : Function.Injective (address s p.acc (D := p.width))) :
    ∃ t, FP.Structural.exec (bitBackend rest cfg exp32) (accumulator Mid Log AccIn Prev Next AccOut p) s = some t ∧
      Post s t AccOut p.acc p.headDim p.width (roundedAccumulator cfg exp32 s Mid Log AccIn Prev Next p) := by
  simpa only [accumulator_software] using
    accumulator_fp32_stage (bitBackend rest cfg exp32) Mid Log AccIn Prev Next AccOut p s hinj

specification sum_software_fp32 [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (exp32 : Word → Word) (Log SumIn Prev Next SumOut : RegionName) (p : StepParams) (s : State Word) :
    ∃ t, FP.Structural.exec (bitBackend rest cfg exp32) (sumStep Log SumIn Prev Next SumOut p) s = some t ∧
      ScalarPost s t SumOut p.sum (roundedSum cfg exp32 s Log SumIn Prev Next p) := by
  simpa only [sum_software] using sum_fp32_stage (bitBackend rest cfg exp32) Log SumIn Prev Next SumOut p s

/-- Supporting recurrence identity only, preserving the old accepted face. -/
specification maximum_recurrence {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (Log : RegionName) (l : FlashDecode2Llama.FPStages.Scalar) (N : Nat) :
    runningMaximum M s Log l 0 = M.negInf ∧
      (∀ k, k < N → runningMaximum M s Log l (k+1) =
        M.binary (some .fp32) .real .max (read M s Log (scalarAddress s l + k))
          (runningMaximum M s Log l k)) := by
  exact ⟨rfl, fun _ _ => rfl⟩

#axiomsClean normalization_fp32_stage
#axiomsClean normalization_software_fp32
#axiomsClean copy_fp32_stage
#axiomsClean copy_software_fp32
#axiomsClean accumulator_fp32_stage
#axiomsClean accumulator_software_fp32
#axiomsClean sum_fp32_stage
#axiomsClean sum_software_fp32
#axiomsClean maximum_recurrence
end
end VeriTile.Bench.TritonBenchG.FlashDecode2Phi.FPCorrect
