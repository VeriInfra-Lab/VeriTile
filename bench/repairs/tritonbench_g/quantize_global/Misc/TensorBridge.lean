import bench.repairs.tritonbench_g.quantize_global.Proofs.KernelCorrectness
import bench.repairs.tritonbench_g.quantize_global.Proofs.InputBinding

/-! Optional original-Tensor application, input binding and check correspondence. -/
namespace QuantizeGlobalRepair.TensorBridge
open VeriTile VeriTile.Triton FP.Structural Direct
noncomputable section

/-- Original caller coordinates and finite-value observation refer to the
same stored words. The absmax buffer is a separate declared kernel input. -/
theorem tensor_kernel_correct (rest : Algebra ℝ)
    (x : Direct.InputTensor) (laws : Direct.InputTensorLaws x) (finite : Bool)
    (h : Direct.InputContract x.metadata finite) (read : Nat → Direct.Word)
    (interpret : Direct.Word → ℝ) (p : Parameters) (length : p.n = x.metadata.numel)
    (v : Invocation) (maximum : ℝ) (initial : State ℝ)
    (binding : Direct.TensorBinding x laws h read interpret p.X initial)
    (finiteBinding : Direct.FiniteObservation x laws read finite)
    (maximumInput : (initial.mem p.Maximum 0).read .real = maximum)
    (scale : KernelCorrectness.ScaleCondition x.metadata.numel
      (fun i => interpret (VeriTile.Torch.FP32.logicalValue read x laws i)) maximum) :
    Spec.Real ((∀ i, i < x.metadata.numel →
      (VeriTile.Torch.FP32.logicalValue read x laws i).toRat?.isSome) ∧
      ∃ final, executeProgram rest Real.arithmetic p v initial = some final ∧
        KernelCorrectness.Result p v
          (fun i => interpret (VeriTile.Torch.FP32.logicalValue read x laws i)) maximum initial final) := by
  refine ⟨Direct.checked_finite x laws finite h read finiteBinding, ?_⟩
  apply KernelCorrectness.correct rest p v _ maximum initial
  · refine ⟨?_, maximumInput⟩
    intro i hi
    exact Direct.original_logical_input x laws h read interpret p.X initial binding i (by simpa [length] using hi)
  · simpa [length] using scale

theorem python_checks_iff_preconditions (x : Direct.ArgumentMetadata) (finite : Bool) :
    Direct.checkInputs x finite = .ok () ↔ Direct.AdmittedArguments x finite :=
  Direct.checkInputs_iff x finite

abbrev original_tensor_binding := @Direct.TensorBinding
abbrev finite_observation := Direct.FiniteObservation

theorem original_logical_input [Inhabited α] (x : InputTensor) (laws : InputTensorLaws x)
    (h : InputContract x.metadata finite) (read : Nat → Word) (interpret : Word → α)
    (X : RegionName) (initial : State α) (binding : TensorBinding x laws h read interpret X initial)
    (i : Nat) (hi : i<x.metadata.numel) :
    inputValues X initial i=interpret (VeriTile.Torch.FP32.logicalValue read x laws i) :=
  Direct.original_logical_input x laws h read interpret X initial binding i hi

theorem original_pointer (x : InputTensor) (laws : InputTensorLaws x)
    (h : InputContract x.metadata finite) (positive : 0<x.metadata.numel) :
    BitVec.ofNat 64 (view x laws h).base=x.dataPtr :=
  Direct.original_pointer x laws h positive

theorem checked_finite (x : InputTensor) (laws : InputTensorLaws x)
    (observed : Bool) (h : InputContract x.metadata observed) (read : Nat → Word)
    (binding : FiniteObservation x laws read observed) (i : Nat) (hi : i < x.metadata.numel) :
    (VeriTile.Torch.FP32.logicalValue read x laws i).toRat?.isSome :=
  Direct.checked_finite x laws observed h read binding i hi

theorem constructed_input_binding [Inhabited α] (x : InputTensor) (laws : InputTensorLaws x)
    (h : InputContract x.metadata finite) (read : Nat → Word) (interpret : Word → α)
    (X : RegionName) (scratch : State α) :
    TensorBinding x laws h read interpret X (bindInput x laws h read interpret X scratch) :=
  Direct.bindInput_binding x laws h read interpret X scratch

#axiomsClean tensor_kernel_correct
#axiomsClean python_checks_iff_preconditions
#axiomsClean original_logical_input
#axiomsClean original_pointer
#axiomsClean checked_finite
#axiomsClean constructed_input_binding
#guard_msgs (drop info) in
#auditModuleAxioms
end
end QuantizeGlobalRepair.TensorBridge
