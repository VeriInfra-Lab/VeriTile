import bench.tritonbench_g.matmul_tma.Sources.Contract
import bench.tritonbench_g.matmul_tma.Sources.Mathematical_FPContract

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticChecked
open VeriTile.Triton FP.Structural FPValue Host
noncomputable section

/-- The existing destination is bound to C; no allocator or returned-object gate. -/
def regions : Regions := ⟨"A","B","C","C"⟩
def params (h : Shape) (f : Flags) : Parameters := parameters h f regions
def kernel (h : Shape) (f : Flags) : ComputeKernel := source (params h f) f.half

@[simp] theorem regionA (h : Shape) (f : Flags) : (params h f).a = "A" := by simp [params,parameters,regions]
@[simp] theorem regionB (h : Shape) (f : Flags) : (params h f).b = "B" := by simp [params,parameters,regions]
@[simp] theorem regionC (h : Shape) (f : Flags) : (params h f).c = "C" := by
  cases hf : f.half <;> simp [params,parameters,regions,returnedTensor,firstOutput,contiguous,hf]

/-- Complete three-block-pointer/two-load/dot/cast/store source execution. -/
theorem executes {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (h : Shape) (f : Flags) (s : State α) :
    ∃ t, exec (backend M D) (kernel h f) s = some t ∧
      t.mem = (scatter (params h f) f.half s (outputValues M D (params h f) f.half s)).mem ∧
      Post (params h f) f.half s t (outputValues M D (params h f) f.half s) := by
  obtain ⟨t,he,hm⟩ := execution_memory M D (params h f) f.half s
  exact ⟨t,he,hm,post_of_memory _ _ _ _ _ (c_injective h f regions) hm⟩

#axiomsClean executes
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticChecked
