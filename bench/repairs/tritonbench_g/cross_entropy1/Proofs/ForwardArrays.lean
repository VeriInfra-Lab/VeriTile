import bench.repairs.tritonbench_g.cross_entropy1.Proofs.TensorOps

namespace VeriTile.Bench.Repairs.CrossEntropy1.Host
open VeriTile.Triton FP.Structural Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 4000000

theorem forward_positive {α σ : Type} [Inhabited α] {M : FP32 α} {rt : Runtime α σ}
    {o : Options} {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    (e : ForwardData M rt o logits labels initial returned final) : 0<e.cols ∧ 0<e.count := by
  have positive : 0<e.cols := by
    by_contra h
    have zero : e.cols=0 := by omega
    apply e.nonzero
    simp [fwdWidth,nextPower,zero]
  exact ⟨positive,block_count_positive e.cols (fwdWidth e.cols) positive (forward_width e.cols positive).1⟩

theorem output_numel (t : Tensor) (count rows : Nat) (positive : 0<count)
    (shape : t.shape=if count>1 then [count,rows] else [rows]) : t.numel=count*rows := by
  by_cases many : count>1
  · simp [Tensor.numel,shape,if_pos many]
  · have one : count=1 := by omega
    simp [Tensor.numel,shape,one]

theorem flattened_slot (count rows index : Nat) (bound : index<count*rows) :
    index%rows<rows ∧ index/rows<count ∧ index/rows*rows+index%rows=index := by
  have positive : 0<rows := by
    by_contra h
    have zero : rows=0 := by omega
    simp [zero] at bound
  exact ⟨Nat.mod_lt index positive,(Nat.div_lt_iff_lt_mul positive).mpr bound,
    by simpa only [Nat.mul_comm rows] using Nat.div_add_mod index rows⟩

def ForwardKernelData.lossValues {α σ : Type} [Inhabited α] {obs : Observations α σ}
    {M : FP32 α} {rt : Runtime α σ} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e) (i : Nat) : α :=
  forwardLoss M e.parameters k.launch.ready.memory (i%e.rows) (i/e.rows)
    (Source.label e.parameters k.launch.ready.memory (i%e.rows))
    (forwardValues M e.parameters k.launch.ready.memory (i%e.rows) (i/e.rows))

def ForwardKernelData.lseValues {α σ : Type} [Inhabited α] {obs : Observations α σ}
    {M : FP32 α} {rt : Runtime α σ} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e) (i : Nat) : α :=
  (forwardValues M e.parameters k.launch.ready.memory (i%e.rows) (i/e.rows)).lse

/-- The actual allocated output views expose exactly the source stores in
block-major order. This includes the empty-row case without any fake reads. -/
theorem forward_arrays {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : Library obs rt) {M : FP32 α} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e) :
    FloatArray obs e.computed e.losses (if e.count>1 then [e.count,e.rows] else [e.rows]) k.lossValues ∧
    FloatArray obs e.computed e.lse (if e.count>1 then [e.count,e.rows] else [e.rows]) k.lseValues := by
  have exit := lib.exit _ _ _ k.launch.exitCall
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
      simpa only [ForwardData.parameters,forwardParameters,base,Nat.zero_add,slot.2.2,
        ForwardKernelData.lossValues] using results.1
    simp only [value,k.lossDense.1,Bool.false_eq_true,↓reduceIte,k.lossDense.2 i hi,stored,Source.Cell.read_same]
  · refine ⟨k.lseAllocation.shapeEq,k.lseAllocation.dtypeEq,exit.live _ k.lseLive,?_⟩
    intro i hi
    have slot := flattened_slot e.count e.rows i (by simpa [lseSize] using hi)
    have results := forward_kernel_outputs obs rt lib k (i%e.rows) (i/e.rows) slot.1 slot.2.1
    have base := dense_base e.lse k.lseDense (by omega)
    have stored : e.computed.memory e.lse.region i=.mk .real (k.lseValues i) := by
      simpa only [ForwardData.parameters,forwardParameters,base,Nat.zero_add,slot.2.2,
        ForwardKernelData.lseValues] using results.2
    simp only [value,k.lseDense.1,Bool.false_eq_true,↓reduceIte,k.lseDense.2 i hi,stored,Source.Cell.read_same]

#axiomsClean forward_arrays
#axiomsClean flattened_slot
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Host
