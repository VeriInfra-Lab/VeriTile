import bench.repairs.tritonbench_g.matmul_triton1.Proofs.References
import bench.repairs.tritonbench_g.matmul_triton1.Proofs.Bytes
import bench.repairs.tritonbench_g.matmul_triton1.revision001.Proofs.Native


namespace VeriTile.Bench.Repairs.MatmulTriton1.TensorBridge
open VeriTile.Triton FP.Structural
open Revision001 Kernel
noncomputable section

/-- Complete selected kernel execution over Real arithmetic, with the
independent full-K mathematical sum at the actual output Tensor addresses.
The same post backs all input/output Tensor ports and both physical frames. -/
theorem matmul_triton1_real (rest : Algebra ℝ) (x : Launch)
    (read : Memory Word) (decode : Word → ℝ) (before : State ℝ)
    (bound : Binding x read decode before) :
    ∃ finite typed post,
      FiniteSource.grid Core.realPrimitives x.parameters before = some finite ∧
      grid rest Core.realPrimitives x before = some typed ∧
      finite.mem = typed.mem ∧ post = image x (decode ∘ read) typed.mem ∧
      Post x before typed (realTarget x read decode) ∧
      Realization x (decode ∘ read) post before typed (realTarget x read decode) := by
  obtain ⟨finite,typed,hf,ht,hm,result⟩ := execute rest Core.realPrimitives x before
  have expected : Post x before typed (realTarget x read decode) := by
    refine ⟨?_,result.2⟩
    intro i
    exact (result.1 i).trans (congrArg (Cell.mk .real) (real_reference x read decode before bound i))
  exact ⟨finite,typed,image x (decode ∘ read) typed.mem,hf,ht,hm,rfl,expected,
    realize x (decode ∘ read) before typed _ (bound.physical x read decode before) expected⟩

#axiomsClean matmul_triton1_real
end
end VeriTile.Bench.Repairs.MatmulTriton1.TensorBridge


namespace VeriTile.Bench.Repairs.MatmulTriton1.TensorBridge
open VeriTile.Triton FP.Structural
open Revision001 Kernel
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- All observations use the same final physical memory, including every
byte of the full input backing allocations and the exterior output frame. -/
structure NativePost (x : Launch) (read post : Memory Word) (before final : State Word)
    (value : Index x → Word) : Prop where
  backing : Realization x read post before final value
  tensor : ∀ i : Index x,
    VeriTile.Torch.FP32.logicalValue post x.c x.lc (offset x i) = value i
  bytes : ∀ (i : Index x) (k : Fin 4),
    observeByte post (address x.c (offset x i)+k.val) = byte (value i) k.val
  inputBytes : ∀ t, t=x.a ∨ t=x.b → ∀ q, storageContains t q →
    observeByte post q = observeByte read q
  byteFrame : ∀ q, q<base x.c ∨ base x.c+4*x.c.metadata.numel ≤ q →
    observeByte post q = observeByte read q

/-- Selected native FP32 scalar FMA order: live carry, ascending K lanes,
and all masked zero padding. No finite-value restriction is imposed. -/
theorem matmul_triton1_native (rest : Algebra Word) (actual : Native.Instructions)
    (x : Launch) (read : Memory Word) (before : State Word) (bound : Binding x read id before) :
    ∃ finite typed post,
      FiniteSource.grid (Native.primitives actual) x.parameters before = some finite ∧
      grid rest (Native.primitives actual) x before = some typed ∧
      finite.mem = typed.mem ∧ post = image x read typed.mem ∧
      Post x before typed (precisionTarget (Native.primitives actual) x read id) ∧
      NativePost x read post before typed (precisionTarget (Native.primitives actual) x read id) := by
  obtain ⟨finite,typed,hf,ht,hm,result⟩ := execute rest (Native.primitives actual) x before
  have expected : Post x before typed (precisionTarget (Native.primitives actual) x read id) := by
    refine ⟨?_,result.2⟩
    intro i
    exact (result.1 i).trans (congrArg (Cell.mk .real)
      (precision_reference (Native.primitives actual) x read id before bound i))
  have physical : PhysicalBinding x read before := bound.physical x read id before
  have realization := realize x read before typed _ physical expected
  refine ⟨finite,typed,image x read typed.mem,hf,ht,hm,rfl,expected,
    realization,?_,output_bytes x read before typed _ expected,
    input_storage_bytes x read typed.mem,byte_frame x read typed.mem⟩
  intro i
  rw [←logicalRead_native _ x.c x.lc x.outputResolved]
  exact realization.outputs i

/-- IEEE software transport needs only the explicit local zero/FMA laws.
This does not assert GPU numerical conformance or FP-to-Real equality. -/
theorem matmul_triton1_fp32_if_local (rest : Algebra Word) (actual : Native.Instructions)
    (numeric : Native.Correspondence actual) (x : Launch) (read : Memory Word)
    (before : State Word) (bound : Binding x read id before) :
    ∃ finite typed post,
      FiniteSource.grid (Native.primitives actual) x.parameters before = some finite ∧
      grid rest (Native.primitives actual) x before = some typed ∧
      finite.mem = typed.mem ∧ post = image x read typed.mem ∧
      Post x before typed (precisionTarget (Native.primitives Native.referenceInstructions) x read id) ∧
      NativePost x read post before typed (precisionTarget (Native.primitives Native.referenceInstructions) x read id) := by
  rw [Native.instruction_eq actual numeric]
  exact matmul_triton1_native rest Native.referenceInstructions x read before bound

#axiomsClean matmul_triton1_native
#axiomsClean matmul_triton1_fp32_if_local
end
end VeriTile.Bench.Repairs.MatmulTriton1.TensorBridge
