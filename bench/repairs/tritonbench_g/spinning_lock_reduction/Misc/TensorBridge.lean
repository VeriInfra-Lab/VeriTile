import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.RawCorrect
import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.RawNative
import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.KernelTensor

/-! Complete checked Tensor applications of both raw phase-kernel results. -/
noncomputable section
namespace VeriTile.Bench.Repairs.SpinningLock.TensorBridge
open VeriTile.Triton VeriTile.Bench.TritonBenchG.SpinningLock
open Admission KernelProof KernelTensor

theorem actual_fp32_logical_read (binding : Binding d t) (checked : tensorCheck d .fp32 device = Bool.true)
    (values : Nat → VeriTile.Torch.FP32.Word) (i : Nat) (hi : i < d.elements) :
    VeriTile.Torch.FP32.logicalValue (wordReader d values) t binding.laws i = values i :=
  KernelTensor.shared_read binding checked values i hi

#axiomsClean actual_fp32_logical_read
end VeriTile.Bench.Repairs.SpinningLock.TensorBridge

noncomputable section
namespace VeriTile.Bench.Repairs.SpinningLock.TensorBridge.Correct
open VeriTile VeriTile.Bench.TritonBenchG.SpinningLock
open Admission KernelProof KernelTensor

open VeriTile.Bench.Repairs.SpinningLock.Correct (mathematical)

theorem producer_kernel_correct (decode : BitVec 32 → ℝ) (zero : decode 0 = 0)
    (ok : check s p c l = Bool.true) :
    Spec.Real (∃ schedule result,
      runKernel realBackend s.args .producers schedule (memory decode p c l) = some result ∧
      Facts realBackend s.args (checkedBuffers ok) .producers schedule (memory decode p c l) result ∧
      (∀ a, result.2.values.mem.output a = mathematical s.args .producers (memory decode p c l) a) ∧
      ConsumerReady s.args result.2.values.mem ∧ Wide.PublishedReady realBackend s.args result.2.values.mem) := by
  obtain ⟨schedule,result,facts⟩ := producer_complete realBackend (by simp [realBackend])
    s.args (checkedBuffers ok) (memory decode p c l) (checked_ready realBackend decode zero ok)
  have ready := producer_supplies_consumer realBackend s.args (checkedBuffers ok) schedule _ result facts
  exact ⟨schedule,result,facts.execution,facts,facts.output,ready.1,ready.2.1⟩

/-- A separately invoked consumer kernel needs the ready producer input.
Fresh non-ready locks remain admitted by the wrapper but do not imply that a
consumer launched before its producers can finish polling. -/
theorem consumer_kernel_correct (decode : BitVec 32 → ℝ) (zero : decode 0 = 0)
    (ok : check s p c l = Bool.true)
    (ready : ConsumerReady s.args (memory decode p c l)) :
    Spec.Real (∃ schedule result,
      runKernel realBackend s.args .consumers schedule (memory decode p c l) = some result ∧
      Facts realBackend s.args (checkedBuffers ok) .consumers schedule (memory decode p c l) result ∧
      ∀ a, result.2.values.mem.output a = mathematical s.args .consumers (memory decode p c l) a) := by
  obtain ⟨schedule,result,facts⟩ := consumer_complete realBackend (by simp [realBackend])
    s.args (checkedBuffers ok) (memory decode p c l) (checked_ready realBackend decode zero ok) ready
  exact ⟨schedule,result,facts.execution,facts,facts.output⟩

/-- Producer postconditions provide exactly the semantic input required by
consumer completion, without modeling wrapper submission or GPU scheduling. -/
theorem producer_to_consumer (ok : check s p c l = Bool.true)
    (facts : Facts realBackend s.args (checkedBuffers ok) .producers schedule mem result) :
    ∃ consumerSchedule consumerResult,
      runKernel realBackend s.args .consumers consumerSchedule result.2.values.mem = some consumerResult ∧
      Facts realBackend s.args (checkedBuffers ok) .consumers consumerSchedule result.2.values.mem consumerResult := by
  have ready := producer_supplies_consumer realBackend s.args (checkedBuffers ok) schedule mem result facts
  obtain ⟨schedule,result,facts⟩ := consumer_complete realBackend (by simp [realBackend])
    s.args (checkedBuffers ok) result.2.values.mem ready.2.1 ready.1
  exact ⟨schedule,result,facts.execution,facts⟩

#axiomsClean producer_kernel_correct
#axiomsClean consumer_kernel_correct
#axiomsClean producer_to_consumer
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.TensorBridge.Correct

noncomputable section
namespace VeriTile.Bench.Repairs.SpinningLock.TensorBridge.FPEquiv
open VeriTile.Triton VeriTile.Bench.TritonBenchG.SpinningLock
open Admission KernelProof KernelTensor

theorem producer_kernel_equiv (ok : check s p c l = Bool.true)
    (pb : Binding p pt) (cb : Binding c ct) (before : ByteHeap)
    (input : InputBytes before p c l) :
    SharedReads p c pt ct pb cb ∧ ∃ schedule result,
      runKernel fp32 s.args .producers schedule (rawFP p c l) = some result ∧
      Facts fp32 s.args (checkedBuffers ok) .producers schedule (rawFP p c l) result ∧
      NativeResult s p c l ct cb .producers before result ∧
      ConsumerReady s.args result.2.values.mem ∧ Wide.PublishedReady fp32 s.args result.2.values.mem := by
  obtain ⟨schedule,result,facts⟩ := producer_complete fp32 fp32_positive_zero_add
    s.args (checkedBuffers ok) (rawFP p c l) (checked_ready fp32 (fun w => ⟨w⟩) rfl ok)
  have ready := producer_supplies_consumer fp32 s.args (checkedBuffers ok) schedule _ result facts
  exact ⟨shared_reads ok pb cb,schedule,result,facts.execution,facts,
    native_result ok cb before input facts,ready.1,ready.2.1⟩

theorem consumer_kernel_equiv (ok : check s p c l = Bool.true)
    (pb : Binding p pt) (cb : Binding c ct) (before : ByteHeap)
    (input : InputBytes before p c l) (ready : ConsumerReady s.args (rawFP p c l)) :
    SharedReads p c pt ct pb cb ∧ ∃ schedule result,
      runKernel fp32 s.args .consumers schedule (rawFP p c l) = some result ∧
      Facts fp32 s.args (checkedBuffers ok) .consumers schedule (rawFP p c l) result ∧
      NativeResult s p c l ct cb .consumers before result := by
  obtain ⟨schedule,result,facts⟩ := consumer_complete fp32 fp32_positive_zero_add
    s.args (checkedBuffers ok) (rawFP p c l) (checked_ready fp32 (fun w => ⟨w⟩) rfl ok) ready
  exact ⟨shared_reads ok pb cb,schedule,result,facts.execution,facts,
    native_result ok cb before input facts⟩

/-- Kernel composition consumes the producer's actual resulting memory. The
consumer's zero accumulator is initialized anew by its own source execution. -/
theorem producer_to_consumer (ok : check s p c l = Bool.true)
    (facts : Facts fp32 s.args (checkedBuffers ok) .producers schedule mem result) :
    ∃ consumerSchedule consumerResult,
      runKernel fp32 s.args .consumers consumerSchedule result.2.values.mem = some consumerResult ∧
      Facts fp32 s.args (checkedBuffers ok) .consumers consumerSchedule result.2.values.mem consumerResult := by
  have ready := producer_supplies_consumer fp32 s.args (checkedBuffers ok) schedule mem result facts
  obtain ⟨schedule,result,facts⟩ := consumer_complete fp32 fp32_positive_zero_add
    s.args (checkedBuffers ok) result.2.values.mem ready.2.1 ready.1
  exact ⟨schedule,result,facts.execution,facts⟩


#axiomsClean producer_kernel_equiv
#axiomsClean consumer_kernel_equiv
#axiomsClean producer_to_consumer
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.TensorBridge.FPEquiv
