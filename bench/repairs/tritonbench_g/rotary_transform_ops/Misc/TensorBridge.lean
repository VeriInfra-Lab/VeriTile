import bench.repairs.tritonbench_g.rotary_transform_ops.Correct
import bench.repairs.tritonbench_g.rotary_transform_ops.Misc.NativeCorrect

/-! Actual Tensor applications and input/metadata/pointer bindings.
Both applications instantiate the exposed raw kernel theorems. -/
namespace VeriTile.Bench.Repairs.RotaryTransformOps.TensorBridge
open VeriTile.Triton FP.Structural KernelEndpoints
noncomputable section

theorem rotary_transform_ops_correct (c : Checked.Call) (tensors : RegionName→KernelTensor.Tensor)
    (binding : KernelTensor.Binding c tensors) (heap : KernelTensor.Heap)
    (observe : Native.Word→ℝ) (other : State ℝ)
    (accepted : Checked.check c (KernelTensor.bytes tensors heap)=Bool.true)
    (pointers : KernelIndex.pointerCheck (Checked.configuration c (KernelTensor.bytes tensors heap)) tensors=Bool.true)
    (suffix : TailReady (Checked.configuration c (KernelTensor.bytes tensors heap))
      (KernelTensor.lift observe c tensors heap other)) :
    let cfg := Checked.configuration c (KernelTensor.bytes tensors heap)
    let initial := KernelTensor.lift observe c tensors heap other
    ∃ final,TypedGrid.run realOps cfg c.host.abi.cumulative64 c.host.abi.offsets64 initial=some final ∧
      (∀ batch row head col,batch<cfg.shape.batches → row<cfg.length batch →
        head<cfg.shape.heads → col<cfg.shape.width →
        (final.mem (KernelAlgebra.regionName cfg.shape.output)
          (cfg.shape.address (cfg.start batch+row) head col)).read .real=
            realFormula cfg (inputMemory observe tensors heap) batch row head col) ∧
      (∀ r o,TypedGrid.Outside cfg r o → final.mem r o=initial.mem r o) := by
  have result := rotary_transform_ops_correct_real _ (Checked.config_valid c _ accepted) _
    (KernelTensor.lift_metadata observe c tensors heap other accepted) suffix
    c.host.abi.cumulative64 c.host.abi.offsets64
  simpa only [real_formula_input] using result

theorem rotary_transform_ops_fp32 (c : Checked.Call) (tensors : RegionName→KernelTensor.Tensor)
    (binding : KernelTensor.Binding c tensors) (heap : KernelTensor.Heap)
    (p : Native.Instructions) (other : State Native.Node)
    (accepted : Checked.check c (KernelTensor.bytes tensors heap)=Bool.true)
    (pointers : KernelIndex.pointerCheck (Checked.configuration c (KernelTensor.bytes tensors heap)) tensors=Bool.true)
    (suffix : TailReady (Checked.configuration c (KernelTensor.bytes tensors heap))
      (KernelTensor.lift Native.atom c tensors heap other)) :
    let cfg := Checked.configuration c (KernelTensor.bytes tensors heap)
    let initial := KernelTensor.lift Native.atom c tensors heap other
    ∃ final,TypedGrid.run (Native.ops p) cfg c.host.abi.cumulative64 c.host.abi.offsets64 initial=some final ∧
      (∀ batch row head col,batch<cfg.shape.batches → row<cfg.length batch →
        head<cfg.shape.heads → col<cfg.shape.width →
        ((final.mem (KernelAlgebra.regionName cfg.shape.output)
          (cfg.shape.address (cfg.start batch+row) head col)).read .real).value=
            nativeFormula cfg p (inputMemory id tensors heap) batch row head col) ∧
      (∀ r o,TypedGrid.Outside cfg r o → final.mem r o=initial.mem r o) := by
  have result := rotary_transform_ops_kernel_fp32 _ (Checked.config_valid c _ accepted) p _
    (KernelTensor.lift_metadata Native.atom c tensors heap other accepted) suffix
    c.host.abi.cumulative64 c.host.abi.offsets64
  simpa only [native_formula_input] using result

alias rotary_transform_ops_input_read := KernelTensor.logical_read
alias rotary_transform_ops_signed_metadata := KernelTensor.lift_metadata
alias rotary_transform_ops_data_pointer := KernelIndex.data_pointer
alias rotary_transform_ops_coefficient_pointer := KernelIndex.coefficient_pointer

#axiomsClean rotary_transform_ops_correct
#axiomsClean rotary_transform_ops_fp32
#axiomsClean rotary_transform_ops_input_read
#axiomsClean rotary_transform_ops_signed_metadata
#axiomsClean rotary_transform_ops_data_pointer
#axiomsClean rotary_transform_ops_coefficient_pointer
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RotaryTransformOps.TensorBridge
