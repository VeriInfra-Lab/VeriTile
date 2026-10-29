import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.Preparation
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.ForwardArrays

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Host
open CrossEntropy1.Source (FP32)
open CrossEntropy1.Host (World Observations FloatArray value fwdWidth block_count_positive
  forward_width output_numel flattened_slot dense_base)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

theorem forward_positive {α σ : Type} [Inhabited α] {M : FP32 α} {rt : Runtime α σ}
    {o : Options} {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    (e : ForwardData M rt o logits labels initial returned final) : 0<e.cols ∧ 0<e.count := by
  have positive := Nat.pos_of_ne_zero e.positive
  exact ⟨positive,block_count_positive e.cols (fwdWidth e.cols) positive (forward_width e.cols positive).1⟩

def ForwardKernelData.lossValues {α σ : Type} [Inhabited α] {obs : Observations α σ}
    {M : FP32 α} {rt : Runtime α σ} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e) (i : Nat) : α :=
  Source.forwardLoss M e.parameters k.launch.ready.memory (i%e.rows) (i/e.rows)
    (CrossEntropy1.Source.label e.parameters.toParameters k.launch.ready.memory (i%e.rows))
    (Source.forwardValues M e.parameters k.launch.ready.memory (i%e.rows) (i/e.rows))

def ForwardKernelData.lseValues {α σ : Type} [Inhabited α] {obs : Observations α σ}
    {M : FP32 α} {rt : Runtime α σ} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e) (i : Nat) : α :=
  (Source.forwardValues M e.parameters k.launch.ready.memory (i%e.rows) (i/e.rows)).lse

/-- The actual allocated output views expose exactly the source stores in
block-major order. This includes the empty-row case without any fake reads. -/
theorem forward_arrays {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : LocalAPI obs rt) {M : FP32 α} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e) :
    FloatArray obs e.computed e.losses (if e.count>1 then [e.count,e.rows] else [e.rows]) k.lossValues ∧
    FloatArray obs e.computed e.lse (if e.count>1 then [e.count,e.rows] else [e.rows]) k.lseValues := by
  have exit := lib.tensor.exit _ _ _ k.launch.exitCall
  have countPositive := (forward_positive e).2
  have lossSize := output_numel e.losses e.count e.rows countPositive k.lossAllocation.shapeEq
  have lseSize := output_numel e.lse e.count e.rows countPositive k.lseAllocation.shapeEq
  constructor
  · refine ⟨k.lossAllocation.shapeEq,k.lossAllocation.dtypeEq,exit.live _ k.lossLive,?_⟩
    intro i hi
    have slot := flattened_slot e.count e.rows i (by simpa [lossSize] using hi)
    have results := forward_kernel_outputs obs rt lib k (i%e.rows) (i/e.rows) slot.1 slot.2.1
    have base := dense_base e.losses k.lossDense (by omega)
    have stored : e.computed.memory e.losses.region i=.mk .real (k.lossValues i) := by
      simpa only [ForwardData.parameters,forwardParameters,CrossEntropy1.Host.forwardParameters,base,Nat.zero_add,slot.2.2,
        ForwardKernelData.lossValues] using results.1
    simp only [value,k.lossDense.1,Bool.false_eq_true,↓reduceIte,k.lossDense.2 i hi,stored,CrossEntropy1.Source.Cell.read_same]
  · refine ⟨k.lseAllocation.shapeEq,k.lseAllocation.dtypeEq,exit.live _ k.lseLive,?_⟩
    intro i hi
    have slot := flattened_slot e.count e.rows i (by simpa [lseSize] using hi)
    have results := forward_kernel_outputs obs rt lib k (i%e.rows) (i/e.rows) slot.1 slot.2.1
    have base := dense_base e.lse k.lseDense (by omega)
    have stored : e.computed.memory e.lse.region i=.mk .real (k.lseValues i) := by
      simpa only [ForwardData.parameters,forwardParameters,CrossEntropy1.Host.forwardParameters,base,Nat.zero_add,slot.2.2,
        ForwardKernelData.lseValues] using results.2.1
    simp only [value,k.lseDense.1,Bool.false_eq_true,↓reduceIte,k.lseDense.2 i hi,stored,CrossEntropy1.Source.Cell.read_same]

#axiomsClean forward_positive
#axiomsClean forward_arrays
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Host
