import bench.tritonbench_g.flash_decode2_llama.Proofs.FPSoftware
import bench.tritonbench_g.flash_decode2_llama.Proofs.FPShape

/-! All previously accepted LLaMA stage-2 value faces with scalar rounding.
The supplied-state stage cut remains explicit; the full online loop is not
claimed. The maximum is the original parsed `tl.maximum` primitive; all three
source projections are checked in `FPShape`. Hardware/compiler conformance
remains pending for every face. -/
namespace VeriTile.Bench.TritonBenchG.FlashDecode2Llama.FPCorrect
open VeriTile.Triton FP.Structural FPStages
noncomputable section

specification normalization_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (Acc SumExp O : RegionName) (a : Row) (z : Scalar) (o : Row) (D : Nat) (s : State α)
    (hinj : Function.Injective (address s o (D := D))) :
    ∃ t, FP.Structural.exec M (normalization Acc SumExp O a z o D) s = some t ∧
      Post s t O o D (normalizationValues M s Acc SumExp a z D) := by
  obtain ⟨t, he, hm⟩ := normalization_memory M Acc SumExp O a z o D s
  exact ⟨t, he, post_of_memory s t O o D _ hinj hm⟩

specification final_copy_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (Final O : RegionName) (a o : Row) (D : Nat) (s : State α)
    (hinj : Function.Injective (address s o (D := D))) :
    ∃ t, FP.Structural.exec M (copy Final O a o D) s = some t ∧
      Post s t O o D (copyValues M s Final a D) := by
  obtain ⟨t, he, hm⟩ := copy_memory M Final O a o D s
  exact ⟨t, he, post_of_memory s t O o D _ hinj hm⟩

specification maximum_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (LogExp Prev Next : RegionName) (block : Nat) (l m : Scalar) (s : State α) :
    ∃ t, FP.Structural.exec M (maximum LogExp Prev Next block l m) s = some t ∧
      ScalarPost s t Next m (maximumValue M s LogExp Prev block l m) := by
  obtain ⟨t, he, hm⟩ := maximum_memory M LogExp Prev Next block l m s
  exact ⟨t, he, scalar_post s t Next m _ hm⟩

specification normalization_software_fp32 [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (Acc SumExp O : RegionName) (a : Row) (z : Scalar) (o : Row) (D : Nat) (s : State Word)
    (hinj : Function.Injective (address s o (D := D))) :
    ∃ t, FP.Structural.exec (bitBackend rest cfg) (normalization Acc SumExp O a z o D) s = some t ∧
      Post s t O o D (roundedNormalization cfg s Acc SumExp a z D) := by
  simpa only [normalization_software] using
    normalization_fp32_stage (bitBackend rest cfg) Acc SumExp O a z o D s hinj

specification final_copy_software_fp32 [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (Final O : RegionName) (a o : Row) (D : Nat) (s : State Word)
    (hinj : Function.Injective (address s o (D := D))) :
    ∃ t, FP.Structural.exec (bitBackend rest cfg) (copy Final O a o D) s = some t ∧
      Post s t O o D (copiedBits s Final a D) := by
  simpa only [copy_software] using final_copy_fp32_stage (bitBackend rest cfg) Final O a o D s hinj

specification maximum_software_fp32 [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (LogExp Prev Next : RegionName) (block : Nat) (l m : Scalar) (s : State Word) :
    ∃ t, FP.Structural.exec (bitBackend rest cfg) (maximum LogExp Prev Next block l m) s = some t ∧
      ScalarPost s t Next m (maximumBits cfg s LogExp Prev block l m) := by
  simpa only [maximum_software] using maximum_fp32_stage (bitBackend rest cfg) LogExp Prev Next block l m s

#axiomsClean normalization_fp32_stage
#axiomsClean final_copy_fp32_stage
#axiomsClean maximum_fp32_stage
#axiomsClean normalization_software_fp32
#axiomsClean final_copy_software_fp32
#axiomsClean maximum_software_fp32
end
end VeriTile.Bench.TritonBenchG.FlashDecode2Llama.FPCorrect
