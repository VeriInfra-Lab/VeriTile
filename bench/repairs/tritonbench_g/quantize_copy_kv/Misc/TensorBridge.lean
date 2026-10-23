import bench.repairs.tritonbench_g.quantize_copy_kv.Correct
import bench.repairs.tritonbench_g.quantize_copy_kv.FPEquiv
import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.KernelWhole
import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.KernelStores

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.TensorBridge
open VeriTile.Triton FP.Structural KernelInputs
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
noncomputable section

/-- Tensor application of the same raw KernelGridResult execution and readback
used by RawKernel.whole, through KernelInputs read and address correspondence. -/
theorem kernel_correct (rest : Algebra ℝ) (m : Inputs) (words : Nat → MetadataWord)
    (h : InputContract m words) (values : Nat → Nat → Nat → ℝ) (initial : State ℝ)
    (reads : RealReads m words values initial) :
    ∃ final,
      KernelGrid.run (KernelExecution.backend (KernelRealModel.algebra rest)
        KernelRealModel.convert KernelRealModel.equal) (parameters m words) Bool.false
        (List.range (request m words).tokens) initial = some final ∧
      KernelGrid.run (KernelExecution.backend (KernelRealModel.algebra rest)
        KernelRealModel.convert KernelRealModel.equal) (parameters m words) Bool.true
        (List.range (request m words).tokens) initial = some final ∧
      KernelWhole.TensorResult m words
        (fun token head column => KernelRealModel.code (values token head) (request m words).columns column)
        (fun token head => KernelRealModel.scale (values token head) (request m words).columns) initial final :=
  KernelWhole.mathematical rest m words h values initial reads

theorem kernel_equiv (a : KernelNativeProfile.Arithmetic) (rest : Algebra Word)
    (m : Inputs) (words : Nat → MetadataWord) (h : InputContract m words)
    (heap : Heap) (binding : TensorBinding m words heap) (initial : State Word)
    (reads : NativeReads m words heap initial) :
    ∃ final,
      KernelGrid.run (KernelNativeProfile.backend a rest) (parameters m words) Bool.false
        (List.range (request m words).tokens) initial = some final ∧
      KernelGrid.run (KernelNativeProfile.backend a rest) (parameters m words) Bool.true
        (List.range (request m words).tokens) initial = some final ∧
      KernelWhole.TensorResult m words (KernelWhole.nativeCode a m words heap)
        (KernelWhole.nativeScale a m words heap) initial final ∧
      KernelStores.NativeStores a m words heap final := by
  obtain ⟨final, original, candidate, result⟩ :=
    KernelWhole.native a rest m words h heap binding initial reads
  exact ⟨final, original, candidate, result,
    KernelStores.of_result a m words h heap binding initial final result⟩

/-- Reused checked INT64 arithmetic gives the same actual input/output byte
addresses, including both FP16 scale bytes, as the shared kernel offsets. -/
alias native_pointer_binding := KernelAddresses.pointers
alias native_metadata_binding := KernelAddresses.destination
alias native_signed_metadata_binding := KernelStores.signed_metadata

#axiomsClean native_pointer_binding
#axiomsClean native_metadata_binding
#axiomsClean native_signed_metadata_binding
#axiomsClean kernel_correct
#axiomsClean kernel_equiv
end
end VeriTile.Bench.Repairs.QuantizeCopyKv.TensorBridge
