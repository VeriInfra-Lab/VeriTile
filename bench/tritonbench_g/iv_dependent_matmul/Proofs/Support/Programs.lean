import bench.tritonbench_g.iv_dependent_matmul.Proofs.IOResults
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.FPEquiv
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue
open IvDependentMatmul
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Existing TF32 K8 instruction model and concrete IEEE conversions. This is
one native graph; no independent numerical rewrite or exact-Real law is assumed. -/
abbrev backend (rest : Algebra Word) (profile : Target.NativeProfile) :=
  KernelNativeResult.backend rest profile

/-- Independent full-K native recurrence, with zero tail lanes, live FP32
carry, final FP16 narrowing and FP32 widening. All FP32 bit patterns remain. -/
def formula (profile : Target.NativeProfile) (g : Kernel.Geometry) := fun vals =>
  IO.result g (Target.native profile.reference g.inner g.blockInner
    (IO.matrixA g vals) (IO.matrixB g vals))

abbrev kernelIO (rest : Algebra Word) (profile : Target.NativeProfile) (g : Kernel.Geometry) :=
  IO.matmulIO (backend rest profile) .accumulator g

/-- Complete native grid on the retained exact FP32 input-cell contract. -/
def original (rest : Algebra Word) (profile : Target.NativeProfile) (g : Kernel.Geometry) :=
  ProgramIO.Implementation.original (kernelIO rest profile g) (fun _ => True) .exactCells

/-- An existing pointer-stage schedule, with the same live-carry graph and IO. -/
def pointerScheduled (rest : Algebra Word) (profile : Target.NativeProfile)
    (g : Kernel.Geometry) (schedule : Kernels.Schedule) :=
  ProgramIO.Implementation.ofVariant (IO.scheduledVariant (backend rest profile) .accumulator g schedule)
    (fun _ => True) .exactCells

end
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.FPEquiv
