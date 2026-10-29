import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Correct
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Misc.NativeCorrect
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.TensorReturns
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.TensorIO
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.LegacyCertificates

/-! Optional actual Tensor applications of the accepted kernel certificates.
`tensors_native` retains the V2 ScalarBinding premise. The nested Unbound
helper preserves the generic V1 statement only as an internal dependency; it
is not an actual-scalar-ABI certificate. -/
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.TensorBridge
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.TritonAttention.Whole001
open FPEquivBound
noncomputable section

/-- Prior operational signatures remain available as ordinary helper theorems. -/
abbrev raw_kernel_real := @LegacyCertificates.kernel_real
abbrev raw_kernel_native := @LegacyCertificates.kernel_native

/- The actual Tensor storage and publication path use the same finite inputs
and complete source schedule as the shared public IO certificates. -/
export TensorStorage (real_io_inputs native_io_inputs real_io_execution native_io_execution)

/-- Actual returned Tensor objects and storage readback for arbitrary Real
input values, using only ordinary Tensor API and fresh-allocation laws. -/
theorem tensors_real (c : Shape) (t : TensorStorage.Objects c) (read : Nat → ℝ)
    (scale : ℝ) (fw pw bw : List Grid.Program)
    (hf : Grid.FullSchedule (Grid.forwardGrid c) fw)
    (hp : Grid.FullSchedule (Grid.preprocessGrid c) pw)
    (hb : Grid.FullSchedule (Grid.backwardGrid c) bw) :
    Spec.Real (∃ r,TensorStorage.invokeReal c t read scale fw pw bw=some r ∧
      TensorStorage.Identities c t r ∧ TensorStorage.RealValues c t read scale r ∧
      (∀ a,¬(∃ region o,Grid.Written c region o ∧ TensorStorage.address c t region o=a) → r.read a=some (read a))) :=
  TensorStorage.real_tensors c t read scale fw pw bw hf hp hb

namespace Unbound
/-- Actual FP32 Tensor object identities, logical storage readback, and frame.
Raw Python integer one selects the constexpr profile; runtime float one
retains scale multiplication and backward FMA. -/
theorem tensors_native (c : Shape) (t : TensorStorage.Objects c) (read : Nat → Native.Word)
    (scalar : Checked.ScalarABI) (owners : TritonAttention.Whole001.Native.Owners)
    (I : Native.Prior.Instructions) (literal : ℝ → Native.Word) (fw pw bw : List Grid.Program)
    (hf : Grid.FullSchedule (Grid.forwardGrid c) fw)
    (hp : Grid.FullSchedule (Grid.preprocessGrid c) pw)
    (hb : Grid.FullSchedule (Grid.backwardGrid c) bw) :
    ∃ r,TensorStorage.invokeNative c t read scalar owners I literal fw pw bw=some r ∧
      TensorStorage.Identities c t r ∧ TensorStorage.NativeValues c t read scalar owners I literal r ∧
      (∀ a,¬(∃ region o,Grid.Written c region o ∧ TensorStorage.address c t region o=a) → r.read a=read a) :=
  TensorStorage.native_tensors c t read scalar owners I literal fw pw bw hf hp hb

#axiomsClean tensors_native
end Unbound

/-- Bound v2 actual-Tensor certificate, with the same local input law as the
kernel endpoint. Returned object identities, logical readback and the full
scalar-address frame are inherited from complete source execution. -/
theorem tensors_native (c : Shape) (t : TensorStorage.Objects c) (read : Nat → Native.Word)
    (scalar : Checked.ScalarABI) (owners : TritonAttention.Whole001.Native.Owners)
    (I : Native.Prior.Instructions) (literal : ℝ → Native.Word)
    (binding : Native.ScalarBinding scalar literal) (fw pw bw : List Grid.Program)
    (hf : Grid.FullSchedule (Grid.forwardGrid c) fw)
    (hp : Grid.FullSchedule (Grid.preprocessGrid c) pw)
    (hb : Grid.FullSchedule (Grid.backwardGrid c) bw) :
    SourceScale scalar I literal ∧
    ∃ r,TensorStorage.invokeNative c t read scalar owners I literal fw pw bw=some r ∧
      TensorStorage.Identities c t r ∧ TensorStorage.NativeValues c t read scalar owners I literal r ∧
      (∀ a,¬(∃ region o,Grid.Written c region o ∧ TensorStorage.address c t region o=a) → r.read a=read a) :=
  ⟨fun kind owner=>binding.source_scale_word scalar literal I kind owner,
    Unbound.tensors_native c t read scalar owners I literal fw pw bw hf hp hb⟩

#axiomsClean tensors_real
#axiomsClean tensors_native
#axiomsClean real_io_inputs
#axiomsClean native_io_inputs
#axiomsClean real_io_execution
#axiomsClean native_io_execution
#axiomsClean TensorStorage.wordInput_logical
#axiomsClean TensorStorage.actual_pointer
#axiomsClean TensorStorage.publish_input
#specNonCircular Native.ScalarBinding avoiding [TensorStorage.invokeNative,Native.Selected.nativeRun,FP.Structural.exec]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.TensorBridge
