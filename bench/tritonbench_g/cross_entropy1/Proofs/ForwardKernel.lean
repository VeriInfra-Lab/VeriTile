import bench.tritonbench_g.cross_entropy1.Proofs.Preparation
import bench.tritonbench_g.cross_entropy1.Proofs.Real

namespace VeriTile.Bench.Repairs.CrossEntropy1.Host
open VeriTile.Triton FP.Structural Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def ForwardData.parameters {α σ : Type} [Inhabited α] {M : FP32 α} {rt : Runtime α σ}
    {o : Options} {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    (e : ForwardData M rt o logits labels initial returned final) : Parameters :=
  forwardParameters o e.workingLogits e.labels e.losses e.lse e.rows e.cols
    (e.world*e.cols) (e.rank*e.cols) (fwdWidth e.cols) (decide (e.world>1 ∨ e.cols>65536))

def ForwardData.count {α σ : Type} [Inhabited α] {M : FP32 α} {rt : Runtime α σ}
    {o : Options} {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    (e : ForwardData M rt o logits labels initial returned final) : Nat := Math.blocks e.cols (fwdWidth e.cols)

theorem dense_base (t : Tensor) (dense : t.Dense) (positive : 0<t.numel) : t.storageOffset=0 := by
  have first := dense.2 0 positive
  unfold Tensor.index at first
  omega

theorem fresh_ne_live {α σ : Type} (obs : Observations α σ) (t : Tensor) (shape : List Nat)
    (dtype : String) (device : Nat) (before after : World α σ)
    (a : Allocation obs t shape dtype device before after) (r : RegionName) (live : before.live r) :
    t.region≠r := by
  intro same
  exact a.fresh (same ▸ live)

/-- Facts derived from the actual preparation, allocation, enter and setup
calls. No separation or numerical kernel output is supplied by the caller. -/
structure ForwardKernelData {α σ : Type} [Inhabited α] (obs : Observations α σ)
    {M : FP32 α} {rt : Runtime α σ} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    (e : ForwardData M rt o logits labels initial returned final) where
  launch : LaunchData rt (rt.setupForward e.parameters (warps e.parameters.block))
    (fun m => Grid.forward M e.parameters m e.count) e.workingLogits.device e.lseAllocated e.computed
  labelsSnapshot : SnapshotValues obs labels e.labels initial e.labelsReady
  working : Materialized obs logits e.workingLogits initial e.resolved
  layout : RowLayout e.workingLogits
  notNeg : e.workingLogits.isNeg=Bool.false
  lossAllocation : Allocation obs e.losses (if e.count>1 then [e.count,e.rows] else [e.rows])
    "torch.float32" e.workingLogits.device e.resolved e.lossAllocated
  lseAllocation : Allocation obs e.lse (if e.count>1 then [e.count,e.rows] else [e.rows])
    "torch.float32" e.workingLogits.device e.lossAllocated e.lseAllocated
  lossDense : e.losses.Dense
  lseDense : e.lse.Dense
  labelsFrame : Preserves obs e.labelsReady launch.ready
  workingFrame : Preserves obs e.resolved launch.ready
  initialFrame : Preserves obs initial launch.ready
  lossLive : launch.ready.live e.losses.region
  lseLive : launch.ready.live e.lse.region
  labelsLive : launch.ready.live e.labels.region
  workingLive : launch.ready.live e.workingLogits.region
  separation : Grid.ForwardSeparation e.parameters

theorem forward_kernel_data {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : Library obs rt) (groups : Groups obs rt) {M : FP32 α} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    (e : ForwardData M rt o logits labels initial returned final)
    (logitsLive : initial.live logits.region) (labelsLive : initial.live labels.region)
    (rank : logits.strides.length=logits.shape.length) : Nonempty (ForwardKernelData obs e) := by
  have prep := forward_preparation obs rt lib groups e logitsLive labelsLive rank
  have loss := lib.empty _ _ _ _ _ _ e.lossCall
  have lse := lib.empty _ _ _ _ _ _ e.lseCall
  obtain ⟨launch⟩ := launch_success rt (rt.setupForward e.parameters (warps e.parameters.block))
    (fun m => Grid.forward M e.parameters m e.count) e.workingLogits.device e.lseAllocated e.computed e.launchCall
  have enter := lib.enter _ _ _ launch.enterCall
  have setup := lib.setupForward _ _ _ _ launch.setupCall
  have prepared := lse.1.frame.trans (enter.trans setup)
  have workingFrame := loss.1.frame.trans prepared
  have labelsFrame := prep.2.2.2.2.trans workingFrame
  have labelsResolved := prep.2.2.2.2.live _ prep.1.materialized.live
  have separation : Grid.ForwardSeparation e.parameters := by
    refine ⟨?_,?_,?_,?_,?_⟩
    · exact fresh_ne_live obs _ _ _ _ _ _ lse.1 _ loss.1.live
    · exact fresh_ne_live obs _ _ _ _ _ _ lse.1 _ (loss.1.frame.live _ prep.2.1.live)
    · exact fresh_ne_live obs _ _ _ _ _ _ loss.1 _ prep.2.1.live
    · simpa only [ForwardData.parameters,forwardParameters,Region.cast_cast] using
        fresh_ne_live obs _ _ _ _ _ _ lse.1 _ (loss.1.frame.live _ labelsResolved)
    · simpa only [ForwardData.parameters,forwardParameters,Region.cast_cast] using
        fresh_ne_live obs _ _ _ _ _ _ loss.1 _ labelsResolved
  exact ⟨⟨launch,prep.1,prep.2.1,prep.2.2.1,prep.2.2.2.1,loss.1,lse.1,loss.2,lse.2,
    labelsFrame,workingFrame,prep.2.1.frame.trans workingFrame,prepared.live _ loss.1.live,
    (enter.trans setup).live _ lse.1.live,labelsFrame.live _ prep.1.materialized.live,
    workingFrame.live _ prep.2.1.live,separation⟩⟩

theorem forward_kernel_logits {α σ : Type} [Inhabited α] (obs : Observations α σ)
    {M : FP32 α} {rt : Runtime α σ} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (dtype : logits.dtype="torch.float32") (row col : Nat) (hr : row<e.rows) (hc : col<e.cols) :
    (k.launch.ready.memory e.parameters.logits (logitAddress e.parameters row col)).read .real=
      value obs initial logits (row*e.cols+col) := by
  have shape : e.workingLogits.shape=[e.rows,e.cols] := k.working.metadata.shape.trans e.logitsShape
  have inside : row*e.cols+col<logits.numel := by
    simp only [Tensor.numel,e.logitsShape,List.prod_cons,List.prod_nil,Nat.mul_one]
    have h := Nat.mul_le_mul_right e.cols (Nat.succ_le_of_lt hr)
    simp only [Nat.succ_mul] at h
    omega
  have logical := k.working.floats dtype _ inside
  rw [← k.workingFrame.floatValue e.workingLogits k.working.live] at logical
  simpa only [value,k.notNeg,Bool.false_eq_true,↓reduceIte,
    k.layout _ _ shape row col hr hc,ForwardData.parameters,forwardParameters,logitAddress] using logical

theorem forward_kernel_labels {α σ : Type} [Inhabited α] (obs : Observations α σ)
    {M : FP32 α} {rt : Runtime α σ} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (dtype : labels.dtype="torch.int64") (row : Nat) (hr : row<e.rows) :
    Source.label e.parameters k.launch.ready.memory row=obs.integer initial labels row := by
  have inputSize : labels.numel=e.rows := by simp [Tensor.numel,e.labelsShape]
  have outputSize : e.labels.numel=e.rows := k.labelsSnapshot.materialized.metadata.numel.trans inputSize
  have base := dense_base e.labels k.labelsSnapshot.dense (by rw [outputSize]; omega)
  have cell := k.labelsSnapshot.labels dtype row (by simpa [inputSize] using hr)
  have frame := k.labelsFrame.memory e.labels.region row k.labelsSnapshot.materialized.live
  simp only [Source.label,ForwardData.parameters,forwardParameters,Region.cast_cast,Region.cast_id,base,Nat.zero_add,frame,cell,Source.Cell.read_same]

/-- The values actually stored by the launched grid survive device cleanup. -/
theorem forward_kernel_outputs {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : Library obs rt) {M : FP32 α} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (row block : Nat) (hr : row<e.rows) (hb : block<e.count) :
    e.computed.memory e.parameters.loss (e.parameters.lossBase+block*e.rows+row)=
      .mk .real (forwardLoss M e.parameters k.launch.ready.memory row block
        (Source.label e.parameters k.launch.ready.memory row)
        (forwardValues M e.parameters k.launch.ready.memory row block)) ∧
    e.computed.memory e.parameters.lse (e.parameters.lseBase+block*e.rows+row)=
      .mk .real (forwardValues M e.parameters k.launch.ready.memory row block).lse := by
  have exit := lib.exit _ _ _ k.launch.exitCall
  have results := Grid.forward_outputs M e.parameters k.launch.ready.memory e.count (row,block)
    ((Grid.task_mem _ _ _).mpr ⟨hr,hb⟩) k.separation
  constructor
  · exact (exit.memory e.parameters.loss _ k.lossLive).trans results.1
  · exact (exit.memory e.parameters.lse _ k.lseLive).trans results.2

#axiomsClean forward_kernel_data
#axiomsClean forward_kernel_logits
#axiomsClean forward_kernel_labels
#axiomsClean forward_kernel_outputs
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Host
