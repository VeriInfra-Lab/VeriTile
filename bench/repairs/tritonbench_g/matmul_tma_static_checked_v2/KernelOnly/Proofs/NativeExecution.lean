import bench.repairs.tritonbench_g.matmul_tma_static_checked_v2.KernelOnly.Proofs.SourceNative
import bench.repairs.tritonbench_g.matmul_tma_static_checked_v2.KernelOnly.Proofs.SourceExecution
import bench.repairs.tritonbench_g.matmul_tma_static_checked_v2.KernelOnly.Kernels
import bench.repairs.tritonbench_g.matmul_tma_static_checked.Proofs.Native

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.NativeExecution
open VeriTile.Triton FP.Structural FPValue Host
open StaticChecked.Native
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Fixed K16 first/carried instructions and final output conversion, applied
only to the raw initial FP16 memory observations. -/
def nativeProduct (I : Instructions) (profile : Profile) (h : Shape) (f : Flags)
    (s : State Word) (i : CIndex (params h f)) : Word :=
  cast I f.half (dot I profile h.inner
    (fun k => aValues (params h f) s (i.1,k,PUnit.unit))
    (fun k => bValues (params h f) s (k,i.2.1,PUnit.unit)))

theorem output_native (rest : Algebra Word) (I : Instructions) (profile : Profile)
    (h : Shape) (f : Flags) (s : State Word) :
    SourceExecution.values (SourceNative.algebra rest I) (sourceDot I profile) (params h f) f.half s=
      nativeProduct I profile h f s := by
  funext i
  cases hf : f.half <;>
    simp only [SourceExecution.values,sourceDot,contraction,SourceNative.algebra,StaticChecked.Native.algebra,nativeProduct,
      outputDType,hf,Bool.false_eq_true,↓reduceIte,StaticChecked.Native.cast] <;> rfl

/-- Complete native source execution and independent initial-memory output,
including the exterior typed-cell frame under the unchanged local ISA laws. -/
theorem kernel_native (rest : Algebra Word) (I : Instructions) (R : Reference)
    (law : Correspondence I R) (profile : Profile) (h : Shape) (f : Flags) (s : State Word) :
    ∃ t,exec (backend (SourceNative.algebra rest I) (sourceDot I profile)) (kernel h f) s=some t ∧
      Post (params h f) f.half s t (nativeProduct (referenceInstructions R) profile h f s) := by
  obtain ⟨t,execution,post⟩ := SourceExecution.correctness (SourceNative.algebra rest I)
    (sourceDot I profile) (params h f) f.half s (c_injective h f StaticChecked.regions)
  rw [output_native] at post
  have equal := instructions_eq I R law
  exact ⟨t,execution,equal ▸ post⟩

#specNonCircular nativeProduct avoiding [kernel,source,FP.Structural.exec,scatter]
#axiomsClean kernel_native
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.NativeExecution
