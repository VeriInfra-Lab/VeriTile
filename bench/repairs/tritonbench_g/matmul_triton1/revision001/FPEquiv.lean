import bench.repairs.tritonbench_g.matmul_triton1.revision001.Proofs.LogicalInputs
import bench.repairs.tritonbench_g.matmul_triton1.revision001.Proofs.PublicContract
import bench.repairs.tritonbench_g.matmul_triton1.revision001.Proofs.Native
namespace VeriTile.Bench.Repairs.MatmulTriton1.Revision001
open VeriTile.Triton FP.Structural
noncomputable section
local instance : Inhabited Native.Word := ⟨FP.Value.zero .fp32⟩

/-- Complete independent software FP32 fused-operation reference. Its native
source interpretation needs the two separately registered local correspondences. -/
specification matmul_triton1_fp_correct (rest : Algebra Native.Word)
    (allocate : Host.Allocator Native.Word) (allocation : Host.AllocationLaw allocate)
    (x : Host.Arguments) (h : Host.Valid x) (initial : State Native.Word)
    (_ha : Host.Storage x.a initial) (_hb : Host.Storage x.b initial) :
    ∃ c final,Host.call rest (Native.primitives Native.referenceInstructions) allocate x initial=some (c,final) ∧
      OutputPost x initial final c
        (fun i => Core.logical (Native.primitives Native.referenceInstructions)
          (Host.parameters x c) initial i.1.val i.2.val) :=
  Host.correct rest (Native.primitives Native.referenceInstructions) allocate allocation x h initial

/-- Native whole-call equality follows from two local numerical conditions:
FP32 zero and one fma.rn.f32 instruction. The exact sequential K schedule is
composed by the proof; no tile, block, full-K or output equality is a premise. -/
specification matmul_triton1_native_correct (rest : Algebra Native.Word)
    (actual : Native.Instructions) (numeric : Native.Correspondence actual)
    (allocate : Host.Allocator Native.Word) (allocation : Host.AllocationLaw allocate)
    (x : Host.Arguments) (h : Host.Valid x) (initial : State Native.Word)
    (_ha : Host.Storage x.a initial) (_hb : Host.Storage x.b initial) :
    ∃ c final,Host.call rest (Native.primitives actual) allocate x initial=some (c,final) ∧
      OutputPost x initial final c
        (fun i => Core.logical (Native.primitives Native.referenceInstructions)
          (Host.parameters x c) initial i.1.val i.2.val) := by
  rw [Native.host_eq actual numeric]
  exact Host.correct rest (Native.primitives Native.referenceInstructions) allocate allocation x h initial

#axiomsClean matmul_triton1_fp_correct
#axiomsClean matmul_triton1_native_correct
/-- Actual source precision, every padded lane and selected instruction
family retained. Native arithmetic is an explicit local port; no FP-to-Real
identity or native numerical conformance is needed for this structural result. -/
specification matmul_triton1_checked_native (rest : Algebra Native.Word) (actual : Native.Instructions)
    (a b : TensorBinding.Tensor) (la : TensorBinding.Laws a) (lb : TensorBinding.Laws b)
    (ra rb : RegionName) (default : Bool) (h : InputContract default a.metadata b.metadata)
    (read : Nat → TensorBinding.Word) (initial : State Native.Word)
    (bound : TensorBinding.Inputs a b la lb h.contiguous.1 h.contiguous.2 ra rb read id initial)
    (allocate : Host.Allocator Native.Word)
    (allocation : CheckedHost.Allocation allocate (InputAdapter.arguments a b ra rb default) initial)
    (launch : CheckedHost.Launchable allocate (InputAdapter.arguments a b ra rb default) initial) :
    ∃ c final, CheckedHost.call (Native.primitives actual) allocate a b ra rb default initial=some (c,final) ∧
      OutputPost (InputAdapter.arguments a b ra rb default) initial final c
        (fun i => LogicalInputs.precisionReference (Native.primitives actual) a b la lb read id
          (InputAdapter.arguments a b ra rb default).config.blockInner i.1.val i.2.val) := by
  obtain ⟨c,final,he,post⟩ := CheckedHost.correct rest (Native.primitives actual) allocate
    a b la lb ra rb default h initial allocation launch
  refine ⟨c,final,he,?_⟩
  have refs := LogicalInputs.precision_reference (Native.primitives actual) a b la lb ra rb default h read id initial bound c
  simpa only [Host.Post,OutputPost,refs] using post

#axiomsClean matmul_triton1_checked_native

/-- Conditional transport from the explicitly stated local instruction laws. -/
specification matmul_triton1_checked_fp32_if_local (rest : Algebra Native.Word) (actual : Native.Instructions)
    (numeric : Native.Correspondence actual)
    (a b : TensorBinding.Tensor) (la : TensorBinding.Laws a) (lb : TensorBinding.Laws b)
    (ra rb : RegionName) (default : Bool) (h : InputContract default a.metadata b.metadata)
    (read : Nat → TensorBinding.Word) (initial : State Native.Word)
    (bound : TensorBinding.Inputs a b la lb h.contiguous.1 h.contiguous.2 ra rb read id initial)
    (allocate : Host.Allocator Native.Word)
    (allocation : CheckedHost.Allocation allocate (InputAdapter.arguments a b ra rb default) initial)
    (launch : CheckedHost.Launchable allocate (InputAdapter.arguments a b ra rb default) initial) :
    ∃ c final, CheckedHost.call (Native.primitives actual) allocate a b ra rb default initial=some (c,final) ∧
      OutputPost (InputAdapter.arguments a b ra rb default) initial final c
        (fun i => LogicalInputs.precisionReference (Native.primitives Native.referenceInstructions) a b la lb read id
          (InputAdapter.arguments a b ra rb default).config.blockInner i.1.val i.2.val) := by
  simpa only [Native.instruction_eq actual numeric] using
    matmul_triton1_checked_native rest Native.referenceInstructions a b la lb ra rb default h
      read initial bound allocate allocation launch

#axiomsClean matmul_triton1_checked_fp32_if_local

end
end VeriTile.Bench.Repairs.MatmulTriton1.Revision001
