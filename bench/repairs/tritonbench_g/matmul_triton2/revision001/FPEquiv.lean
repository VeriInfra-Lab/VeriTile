import bench.repairs.tritonbench_g.matmul_triton2.revision001.Proofs.Autotune
import bench.repairs.tritonbench_g.matmul_triton2.revision001.Proofs.LogicalInputs
import bench.repairs.tritonbench_g.matmul_triton2.revision001.Contract
import bench.repairs.tritonbench_g.matmul_triton2.revision001.Proofs.PublicContract
import bench.repairs.tritonbench_g.matmul_triton2.revision001.Proofs.Native
namespace VeriTile.Bench.Repairs.MatmulTriton2.Revision001
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulTriton1.Revision001
noncomputable section
local instance : Inhabited Native.Word := ⟨FP.Value.zero .fp32⟩

/-- Source-selected fixed-k8 instruction graph over FP32 words. Actual MMA
numerical semantics are local ports, never a whole-dot/output assumption. -/
specification matmul_triton2_native_execution (rest : Algebra Native.Word)
    (actual : Native.Instructions) (choice : Fin 8)
    (allocate : Host.Allocator Native.Word) (allocation : Host.AllocationLaw allocate)
    (x : Host.Arguments) (_selected : x.config=Native.config choice)
    (h : Host.Valid x) (initial : State Native.Word)
    (_ha : Host.Storage x.a initial) (_hb : Host.Storage x.b initial) :
    ∃ c final,Host.call rest (Native.primitives actual (Native.family choice)) allocate x initial=some (c,final) ∧
      OutputPost x initial final c
        (fun i => Core.logical (Native.primitives actual (Native.family choice))
          (Host.parameters x c) initial i.1.val i.2.val) :=
  Host.correct rest (Native.primitives actual (Native.family choice)) allocate allocation x h initial

/-- Three local conditions: FP32 zero, fixed-arity sync MMA8, and fixed-arity
warp-group MMA8. The reference ports must describe the selected TF32-input,
FP32-seed/result hardware operations; their native conformance remains deferred.
Whole-call success, all output values, return identity and frame are derived. -/
specification matmul_triton2_fp_correct (rest : Algebra Native.Word)
    (actual : Native.Instructions) (reference : Native.Reference)
    (numeric : Native.Correspondence actual reference) (choice : Fin 8)
    (allocate : Host.Allocator Native.Word) (allocation : Host.AllocationLaw allocate)
    (x : Host.Arguments) (_selected : x.config=Native.config choice)
    (h : Host.Valid x) (initial : State Native.Word)
    (_ha : Host.Storage x.a initial) (_hb : Host.Storage x.b initial) :
    ∃ c final,Host.call rest (Native.primitives actual (Native.family choice)) allocate x initial=some (c,final) ∧
      OutputPost x initial final c
        (fun i => Core.logical (Native.primitives (Native.referenceInstructions reference) (Native.family choice))
          (Host.parameters x c) initial i.1.val i.2.val) := by
  rw [Native.host_eq actual reference numeric]
  exact Host.correct rest (Native.primitives (Native.referenceInstructions reference) (Native.family choice))
    allocate allocation x h initial

#axiomsClean matmul_triton2_native_execution
#axiomsClean matmul_triton2_fp_correct
/-- Actual source precision, every padded lane and selected instruction
family retained. Native arithmetic is an explicit local port; no FP-to-Real
identity or native numerical conformance is needed for this structural result. -/
specification matmul_triton2_checked_native (rest : Algebra Native.Word) (actual : Native.Instructions)
    (a b : TensorBinding.Tensor) (la : TensorBinding.Laws a) (lb : TensorBinding.Laws b)
    (ra rb : RegionName) (choice : Fin 8) (h : InputContract a.metadata b.metadata)
    (read : Nat → TensorBinding.Word) (initial : State Native.Word)
    (bound : TensorBinding.Inputs a b la lb h.contiguous.1 h.contiguous.2 ra rb read id initial)
    (allocate : Host.Allocator Native.Word)
    (allocation : CheckedHost.Allocation allocate (InputAdapter.arguments a b ra rb choice) initial)
    (launch : CheckedHost.Launchable allocate (InputAdapter.arguments a b ra rb choice) initial) :
    ∃ c final, CheckedHost.call (Native.primitives actual (Native.family choice)) allocate a b ra rb choice initial=some (c,final) ∧
      OutputPost (InputAdapter.arguments a b ra rb choice) initial final c
        (fun i => LogicalInputs.precisionReference (Native.primitives actual (Native.family choice)) a b la lb read id
          (InputAdapter.arguments a b ra rb choice).config.blockInner i.1.val i.2.val) := by
  obtain ⟨c,final,he,post⟩ := CheckedHost.correct rest (Native.primitives actual (Native.family choice)) allocate
    a b la lb ra rb choice h initial allocation launch
  refine ⟨c,final,he,?_⟩
  have refs := LogicalInputs.precision_reference (Native.primitives actual (Native.family choice)) a b la lb ra rb choice h read id initial bound c
  simpa only [Host.Post,OutputPost,refs] using post

#axiomsClean matmul_triton2_checked_native

/-- Complete tuner trial sequence and final selected launch. -/
specification matmul_triton2_autotuned_native (rest : Algebra Native.Word) (actual : Native.Instructions)
    (a b : TensorBinding.Tensor) (la : TensorBinding.Laws a) (lb : TensorBinding.Laws b)
    (ra rb : RegionName) (plan : Autotune.Plan) (h : InputContract a.metadata b.metadata)
    (read : Nat → TensorBinding.Word) (initial : State Native.Word)
    (bound : TensorBinding.Inputs a b la lb h.contiguous.1 h.contiguous.2 ra rb read id initial)
    (allocate : Host.Allocator Native.Word)
    (allocation : CheckedHost.Allocation allocate (InputAdapter.arguments a b ra rb plan.selected) initial)
    (launch : Autotune.Launchable allocate a b ra rb plan initial) :
    ∃ c final, Autotune.call (fun choice => Native.primitives actual (Native.family choice)) allocate a b ra rb plan initial=some (c,final) ∧
      OutputPost (InputAdapter.arguments a b ra rb plan.selected) initial final c
        (fun i => LogicalInputs.precisionReference (Native.primitives actual (Native.family plan.selected)) a b la lb read id
          (InputAdapter.arguments a b ra rb plan.selected).config.blockInner i.1.val i.2.val) := by
  obtain ⟨c,final,he,post⟩ := Autotune.correct rest (fun choice => Native.primitives actual (Native.family choice)) allocate
    a b la lb ra rb plan h initial allocation launch
  refine ⟨c,final,he,?_⟩
  have refs := LogicalInputs.precision_reference (Native.primitives actual (Native.family plan.selected)) a b la lb ra rb plan.selected h read id initial bound c
  simpa only [Host.Post,OutputPost,refs] using post

#axiomsClean matmul_triton2_autotuned_native

/-- Conditional transport from the explicitly stated local instruction laws. -/
specification matmul_triton2_autotuned_tf32_if_local (rest : Algebra Native.Word) (actual : Native.Instructions)
    (reference : Native.Reference) (numeric : Native.Correspondence actual reference)
    (a b : TensorBinding.Tensor) (la : TensorBinding.Laws a) (lb : TensorBinding.Laws b)
    (ra rb : RegionName) (plan : Autotune.Plan) (h : InputContract a.metadata b.metadata)
    (read : Nat → TensorBinding.Word) (initial : State Native.Word)
    (bound : TensorBinding.Inputs a b la lb h.contiguous.1 h.contiguous.2 ra rb read id initial)
    (allocate : Host.Allocator Native.Word)
    (allocation : CheckedHost.Allocation allocate (InputAdapter.arguments a b ra rb plan.selected) initial)
    (launch : Autotune.Launchable allocate a b ra rb plan initial) :
    ∃ c final, Autotune.call (fun choice => Native.primitives actual (Native.family choice)) allocate a b ra rb plan initial=some (c,final) ∧
      OutputPost (InputAdapter.arguments a b ra rb plan.selected) initial final c
        (fun i => LogicalInputs.precisionReference (Native.primitives (Native.referenceInstructions reference) (Native.family plan.selected)) a b la lb read id
          (InputAdapter.arguments a b ra rb plan.selected).config.blockInner i.1.val i.2.val) := by
  have same : (fun choice => Native.primitives actual (Native.family choice)) =
      (fun choice => Native.primitives (Native.referenceInstructions reference) (Native.family choice)) := by
    funext choice
    exact Native.primitives_eq actual reference numeric (Native.family choice)
  simpa only [same] using
    matmul_triton2_autotuned_native rest (Native.referenceInstructions reference) a b la lb ra rb plan h
      read initial bound allocate allocation launch

#axiomsClean matmul_triton2_autotuned_tf32_if_local

end
end VeriTile.Bench.Repairs.MatmulTriton2.Revision001
