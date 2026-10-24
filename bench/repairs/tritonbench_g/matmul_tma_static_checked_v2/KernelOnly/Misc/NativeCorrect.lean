import bench.repairs.tritonbench_g.matmul_tma_static_checked_v2.KernelOnly.Proofs.PublicIO
import bench.repairs.tritonbench_g.matmul_tma_static_checked_v2.KernelOnly.Proofs.NativeExecution

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2
open VeriTile.Triton FP.Structural FPValue Host
open StaticChecked.Native
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Independent fixed K16 first/carried recurrence and final output conversion
on logical input matrices, with the selected instruction profile retained. -/
def nativeProduct (I : Instructions) (profile : Profile) (h : Shape) (f : Flags)
    (a : AIndex (params h f) → Word) (b : BIndex (params h f) → Word)
    (i : CIndex (params h f)) : Word :=
  cast I f.half (dot I profile h.inner (fun k => a (i.1,k,PUnit.unit))
    (fun k => b (k,i.2.1,PUnit.unit)))

/-- Actual FP16 input channels, stride selection, complete source and output
conversion. Primitive instructions are model parameters, not experiments. -/
def matmulNativeIO (rest : Algebra Word) (I : Instructions) (profile : Profile)
    (h : Shape) (f : Flags) : ProgramIO Word Word :=
  PublicIO.io (StaticChecked.Native.algebra rest I) (sourceDot I profile) (params h f) f.half

/-- Successful whole-source native observation and exact frame under the
unchanged local first/carry/conversion correspondence. -/
theorem kernel_native (rest : Algebra Word) (I : Instructions) (R : Reference)
    (law : Correspondence I R) (profile : Profile) (h : Shape) (f : Flags) :
    (matmulNativeIO rest I profile h f).Implements
      (PublicIO.formula (params h f) f.half
        (nativeProduct (referenceInstructions R) profile h f)) := by
  rw [instructions_eq I R law]
  refine PublicIO.implements (StaticChecked.Native.algebra rest (referenceInstructions R))
    (sourceDot (referenceInstructions R) profile) (params h f) f.half
    (c_injective h f StaticChecked.regions) _ ?_
  intro a b i
  cases hf : f.half <;>
    simp only [StaticChecked.Native.algebra, sourceDot, contraction, nativeProduct,
      outputDType, hf, Bool.false_eq_true, ↓reduceIte, StaticChecked.Native.cast] <;> rfl

#specNonCircular nativeProduct avoiding [kernel, source, FP.Structural.exec, scatter]
#axiomsClean kernel_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2
