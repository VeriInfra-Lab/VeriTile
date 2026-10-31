import bench.tritonbench_g.lora_expand_gemv.Sources.Mathematical_BranchSource

namespace VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches
open VeriTile.Triton
open FPValue (OutputFormat)
noncomputable section
set_option maxHeartbeats 40000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false

theorem sourceAlgorithmSupported (profile : InputProfile) (out : OutputFormat) (p : Parameters) :
    ∃ alg, (source profile out p).toAlgorithm? = Except.ok alg := by
  cases profile <;> cases out
  all_goals exact ⟨_, rfl⟩

#axiomsClean sourceAlgorithmSupported
end
end VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches
