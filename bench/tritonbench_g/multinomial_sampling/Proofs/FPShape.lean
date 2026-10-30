import bench.tritonbench_g.multinomial_sampling.Proofs.FPSource
import bench.tests.FPMultinomialSource
namespace VeriTile.Bench.TritonBenchG.MultinomialSampling.FPValue
open VeriTile.Triton

theorem fp32_source_exact : original=VeriTile.Tests.FPMultinomialSource.original := rfl
theorem fp16_source_exact : original16=VeriTile.Tests.FPMultinomialSource.original16 := rfl
theorem bf16_source_exact : originalBfloat16=VeriTile.Tests.FPMultinomialSource.originalBfloat16 := rfl
theorem fp64_source_exact : original64=VeriTile.Tests.FPMultinomialSource.original64 := rfl

theorem full_body_lengths (d : Storage) (p : Parameters) :
    (prologue d p).length=11 ∧ (body d p).length=12 ∧ (epilogue d p).length=1 := by
  cases d <;> exact ⟨rfl,rfl,rfl⟩

#axiomsClean fp32_source_exact
#axiomsClean fp16_source_exact
#axiomsClean bf16_source_exact
#axiomsClean fp64_source_exact
#axiomsClean full_body_lengths
end VeriTile.Bench.TritonBenchG.MultinomialSampling.FPValue
