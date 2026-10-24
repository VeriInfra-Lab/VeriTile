import bench.repairs.tritonbench_g.matmul_triton1.Proofs.KernelInterface
import bench.repairs.tritonbench_g.matmul_triton1.revision001.Proofs.Native

namespace VeriTile.Bench.Repairs.MatmulTriton1.KernelNativeResults
open VeriTile.Triton FP.Structural
open Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
local instance : Inhabited Native.Word := ⟨FP.Value.zero .fp32⟩

/-- Native full-grid source execution, ordered FMA target and complete frames. -/
theorem matmul_triton1_native (rest : Algebra Native.Word) (actual : Native.Instructions)
    (p : Core.Parameters) (canonical : FiniteSource.Canonical p)
    (ready : FiniteIndex.Ready p) (storage : FiniteIndex.StorageBounds p)
    (geometry : Geometry.Valid (Core.geometry p))
    (ha : p.input ≠ p.output) (hb : p.weight ≠ p.output) (initial : State Native.Word) :
    ∃ finite typed, FiniteSource.grid (Native.primitives actual) p initial = some finite ∧
      Core.grid rest (Native.primitives actual) p initial = some typed ∧ finite.mem = typed.mem ∧
      RawKernel.Post p initial finite (fun i => Core.logical (Native.primitives actual) p initial i.1.val i.2.val) ∧
      RawKernel.Post p initial typed (fun i => Core.logical (Native.primitives actual) p initial i.1.val i.2.val) :=
  RawKernel.execute rest (Native.primitives actual) p canonical ready storage geometry ha hb initial

/-- The same explicitly conditional local zero/FMA transport, on raw states. -/
theorem matmul_triton1_fp32_if_local (rest : Algebra Native.Word) (actual : Native.Instructions)
    (numeric : Native.Correspondence actual) (p : Core.Parameters) (canonical : FiniteSource.Canonical p)
    (ready : FiniteIndex.Ready p) (storage : FiniteIndex.StorageBounds p)
    (geometry : Geometry.Valid (Core.geometry p))
    (ha : p.input ≠ p.output) (hb : p.weight ≠ p.output) (initial : State Native.Word) :
    ∃ finite typed, FiniteSource.grid (Native.primitives actual) p initial = some finite ∧
      Core.grid rest (Native.primitives actual) p initial = some typed ∧ finite.mem = typed.mem ∧
      RawKernel.Post p initial finite (fun i => Core.logical (Native.primitives Native.referenceInstructions) p initial i.1.val i.2.val) ∧
      RawKernel.Post p initial typed (fun i => Core.logical (Native.primitives Native.referenceInstructions) p initial i.1.val i.2.val) := by
  rw [Native.instruction_eq actual numeric]
  exact matmul_triton1_native rest Native.referenceInstructions p canonical ready storage geometry ha hb initial

#axiomsClean matmul_triton1_native
#axiomsClean matmul_triton1_fp32_if_local
end
end VeriTile.Bench.Repairs.MatmulTriton1.KernelNativeResults
