import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.KernelStages
import bench.tritonbench_g.spinning_lock_reduction.Proofs.FP

/-! Complete raw-state producer and consumer kernel results. Facts includes successful
source execution, independent output values, reference replay and all memory frames. -/
noncomputable section
namespace VeriTile.Bench.Repairs.SpinningLock.FPEquiv
open VeriTile VeriTile.Triton VeriTile.Bench.TritonBenchG.SpinningLock
open KernelProof

theorem producer_kernel_equiv_raw (h : HostArgs) (buf : Wide.Buffers h)
    (mem : Memory (FP.Value .fp32)) (published : Wide.PublishedReady fp32 h mem) :
    ∃ schedule result,
      Facts fp32 h buf .producers schedule mem result ∧
      ConsumerReady h result.2.values.mem ∧ Wide.PublishedReady fp32 h result.2.values.mem := by
  obtain ⟨schedule,result,facts⟩ := producer_complete fp32 fp32_positive_zero_add h buf mem published
  have ready := producer_supplies_consumer fp32 h buf schedule mem result facts
  exact ⟨schedule,result,facts,ready.1,ready.2.1⟩

theorem consumer_kernel_equiv_raw (h : HostArgs) (buf : Wide.Buffers h)
    (mem : Memory (FP.Value .fp32)) (published : Wide.PublishedReady fp32 h mem)
    (ready : ConsumerReady h mem) :
    ∃ schedule result, Facts fp32 h buf .consumers schedule mem result :=
  consumer_complete fp32 fp32_positive_zero_add h buf mem published ready

theorem producer_to_consumer_raw (h : HostArgs) (buf : Wide.Buffers h)
    (facts : Facts fp32 h buf .producers schedule mem result) :
    ∃ consumerSchedule consumerResult,
      runKernel fp32 h .consumers consumerSchedule result.2.values.mem = some consumerResult ∧
      Facts fp32 h buf .consumers consumerSchedule result.2.values.mem consumerResult := by
  have ready := producer_supplies_consumer fp32 h buf schedule mem result facts
  obtain ⟨schedule,result,facts⟩ := consumer_complete fp32 fp32_positive_zero_add
    h buf result.2.values.mem ready.2.1 ready.1
  exact ⟨schedule,result,facts.execution,facts⟩

alias exact_source_execution := SourceReader.runKernel_projection

#axiomsClean producer_kernel_equiv_raw
#axiomsClean consumer_kernel_equiv_raw
#axiomsClean producer_to_consumer_raw
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.FPEquiv
