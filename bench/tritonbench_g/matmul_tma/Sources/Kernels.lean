import bench.tritonbench_g.matmul_tma.Sources.Contract
import bench.tritonbench_g.matmul_tma.Sources.Mathematical_FPContract

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticChecked
open VeriTile.Triton FP.Structural FPValue Host
noncomputable section

/-- The existing destination is bound to C; no allocator or returned-object gate. -/
def regions : Regions := ⟨"A","B","C","C"⟩
def params (h : Shape) (f : Flags) : Parameters := parameters h f regions

@[simp] theorem regionA (h : Shape) (f : Flags) : (params h f).a = "A" := by simp [params,parameters,regions]
@[simp] theorem regionB (h : Shape) (f : Flags) : (params h f).b = "B" := by simp [params,parameters,regions]
@[simp] theorem regionC (h : Shape) (f : Flags) : (params h f).c = "C" := by
  cases hf : f.half <;> simp [params,parameters,regions,returnedTensor,firstOutput,contiguous,hf]

end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticChecked
