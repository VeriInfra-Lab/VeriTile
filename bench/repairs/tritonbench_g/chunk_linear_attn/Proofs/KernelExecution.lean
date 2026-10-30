import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.ReferenceOutput
import bench.repairs.tritonbench_g.chunk_linear_attn.BoundMeaningGrid
import bench.repairs.tritonbench_g.chunk_linear_attn.NativeFP

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.KernelExecution
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 8000000
variable {α : Type} [Inhabited α]

/-- Kernel launch composition only. Each grid executes the shared quotations;
the final finite scalar tree is the existing library reduction boundary. -/
def run (M : Algebra α) (D : DotPrimitive α) (c : Real.Call) (input : Memory α) : Option (Memory α) := do
  let kernels ← executeFour M D c.forward c.backward c.scale c.seed c.final c.terminal
    (grid .forwardState c.forward) (grid .forwardOutput c.forward)
    (grid .backwardState c.backward) (grid .backwardGradient c.backward) input
  return hostSumMemory M c.backward c.selection kernels kernels

theorem source_complete (M : Algebra α) (D : DotPrimitive α) (c : Real.Call) (input : Memory α) :
    run M D c input=some (completeMemory M D c.forward c.backward c.scale
      c.seed c.final c.terminal c.selection input) := by
  simp only [run,four_execution,bind,Option.bind_some,completeMemory]
  rfl

theorem reference_complete (M : Algebra α) (D : DotPrimitive α) (c : Real.Call)
    (valid : Host.Valid c.dims) (input : Memory α) :
    run M D c input=some (KernelReference.result M D c.dims valid c.scale
      c.seed c.final c.terminal c.selection input) := by
  rw [source_complete]
  exact congrArg some (KernelReference.result_eq M D c.dims valid c.scale c.seed c.final c.terminal c.selection input)

/-- The native profile is an explicit fixed scalar/FMA interpretation. Runtime
scale uses the actual ABI word; only the source's scale placeholder is bound.
This retains all source casts and load boundaries and exceptional words. -/
def project (profile : NativeTerms.FP32Profile) (c : Real.Call) (scale : BitVec 32)
    (symbolic : Memory NativeTerms.Term) : RegionName → Nat → BitVec 32 := fun r o =>
  (NativeTerms.lowerAt c.backward r (NativeBound.bindScale scale ((symbolic r o).read .real))).eval profile.isa

def symbolicCall (c : Real.Call) : Real.Call := {c with scale:=1}

def runNative (profile : NativeTerms.FP32Profile) (c : Real.Call) (scale : BitVec 32)
    (input : RegionName → Nat → BitVec 32) : Option (RegionName → Nat → BitVec 32) := do
  let symbolic ← run NativeTerms.algebra NativeTerms.dotPrimitive (symbolicCall c) (NativeTerms.inputMemory input)
  return project profile c scale symbolic

/-- Independent coordinate-table native target. Its scalar instruction profile
matches source projection by definition; no compiler/lowering equality is a
premise. Interpretation by actual CUDA is an explicit compiler/ISA boundary. -/
def nativeTarget (profile : NativeTerms.FP32Profile) (c : Real.Call) (valid : Host.Valid c.dims)
    (scale : BitVec 32) (input : RegionName → Nat → BitVec 32) : RegionName → Nat → BitVec 32 :=
  project profile c scale (KernelReference.result NativeTerms.algebra NativeTerms.dotPrimitive
    c.dims valid 1 c.seed c.final c.terminal c.selection (NativeTerms.inputMemory input))

theorem native_complete (profile : NativeTerms.FP32Profile) (c : Real.Call) (valid : Host.Valid c.dims)
    (scale : BitVec 32) (input : RegionName → Nat → BitVec 32) :
    runNative profile c scale input=some (nativeTarget profile c valid scale input) := by
  unfold runNative
  rw [reference_complete NativeTerms.algebra NativeTerms.dotPrimitive (symbolicCall c) valid]
  rfl

#axiomsClean source_complete
#axiomsClean reference_complete
#axiomsClean native_complete
#specNonCircular nativeTarget avoiding [run,runNative,FP.Structural.exec,runGrid,executeFour,
  gridMemory,stageMemory,modelFour,completeMemory,NativeTerms.rawMemory,NativeBound.rawMemory]
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.KernelExecution
