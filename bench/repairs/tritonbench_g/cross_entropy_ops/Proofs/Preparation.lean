import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.Trace
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.ForwardKernel
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.TensorOps

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Host
open VeriTile.Triton
open CrossEntropy1.Source (FP32 Memory Cell)
open CrossEntropy1.Host (World Step Observations Preserves Materialized SnapshotValues RowLayout
  Allocation Groups Library rowStride lastStride fwdWidth bwdWidth warps value
  normalization_values snapshot_values fresh_ne_live dense_base)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

/-- Only the additional ordinary effects of the Ops wrapper. Common tensor,
allocator, device and saved-object contracts are reused without alteration. -/
structure LocalAPI {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ) : Prop where
  tensor : Library obs rt.toRuntime
  setupForward : ∀ p n before after, rt.setupForwardOps p n before=.ok () after → Preserves obs before after
  setupBackward : ∀ p n before after, rt.setupBackwardOps p n before=.ok () after → Preserves obs before after
  mark : ∀ t before after, rt.markNonDifferentiable t before=.ok () after → Preserves obs before after

structure ZeroAPI {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ) (zero : α) : Prop where
  zeros : ∀ input before output after, before.live input.region → input.dtype="torch.float32" →
    rt.zerosLike input before=.ok output after →
    CrossEntropy1.Host.Produced obs before after output input.shape input.device (fun _ => zero)

theorem group_metadata {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (groups : Groups obs rt.toRuntime) (group : Option Nat) (initial sized ranked : World α σ) (world rank : Nat)
    (sizeCall : CrossEntropy1.Host.groupSize rt.toRuntime group initial=.ok world sized)
    (rankCall : CrossEntropy1.Host.groupRank rt.toRuntime group sized=.ok rank ranked) :
    0<world ∧ rank<world ∧ Preserves obs initial ranked := by
  cases hgroup : group with
  | none =>
    simp only [CrossEntropy1.Host.groupSize,hgroup,Step.ok.injEq] at sizeCall
    simp only [CrossEntropy1.Host.groupRank,hgroup,Step.ok.injEq] at rankCall
    have frame : Preserves obs initial ranked := by
      rw [← rankCall.2,← sizeCall.2]
      exact Preserves.refl obs initial
    exact ⟨by omega,by omega,frame⟩
  | some g =>
    simp only [CrossEntropy1.Host.groupSize,hgroup] at sizeCall
    simp only [CrossEntropy1.Host.groupRank,hgroup] at rankCall
    have sizes := groups.sizeCall _ _ _ _ sizeCall
    have ranks := groups.rankCall _ _ _ _ rankCall
    exact ⟨by simpa [sizes.1] using groups.positive g,
      by simpa [sizes.1,ranks.1] using groups.rankBound g,sizes.2.trans ranks.2⟩

theorem forward_preparation {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : LocalAPI obs rt) (groups : Groups obs rt.toRuntime) {M : FP32 α} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    (e : ForwardData M rt o logits labels initial returned final)
    (logitsLive : initial.live logits.region) (labelsLive : initial.live labels.region)
    (rank : logits.strides.length=logits.shape.length) :
    SnapshotValues obs labels e.labels initial e.labelsReady ∧
    Materialized obs logits e.workingLogits initial e.resolved ∧ RowLayout e.workingLogits ∧
      e.workingLogits.isNeg=Bool.false ∧ Preserves obs e.labelsReady e.resolved := by
  have groupData := group_metadata obs rt groups o.processGroup initial e.sized e.ranked e.world e.rank e.sizeCall e.rankCall
  have snapshot := snapshot_values obs rt.toRuntime lib.tensor labels e.labels e.ranked e.labelsReady
    (groupData.2.2.live _ labelsLive) e.snapshotCall
  have labelsBefore := snapshot.prepend groupData.2.2 labelsLive
  have normalized := normalization_values obs rt.toRuntime lib.tensor logits e.inplaceLogits e.labelsReady e.normalized
    (labelsBefore.materialized.frame.live _ logitsLive) rank e.originalNormalize
  have original := normalized.1.prepend labelsBefore.materialized.frame logitsLive
  have resolved := lib.tensor.resolve _ _ _ _ original.live e.resolveCall
  have working := normalization_values obs rt.toRuntime lib.tensor e.resolvedLogits e.workingLogits e.resolvedState e.resolved
    resolved.1.live resolved.1.rank e.workingNormalize
  exact ⟨labelsBefore,(original.trans resolved.1).trans working.1,working.2.1,working.2.2 resolved.2,
    normalized.1.frame.trans (resolved.1.frame.trans working.1.frame)⟩

def ForwardData.parameters {α σ : Type} [Inhabited α] {M : FP32 α} {rt : Runtime α σ}
    {o : Options} {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    (e : ForwardData M rt o logits labels initial returned final) : Parameters :=
  forwardParameters o e.workingLogits e.labels e.losses e.lse e.zlosses e.rows e.cols
    (e.world*e.cols) (e.rank*e.cols) (fwdWidth e.cols) (decide (e.world>1 ∨ e.cols>65536))

def ForwardData.count {α σ : Type} [Inhabited α] {M : FP32 α} {rt : Runtime α σ}
    {o : Options} {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    (e : ForwardData M rt o logits labels initial returned final) : Nat := CrossEntropy1.Math.blocks e.cols (fwdWidth e.cols)

structure ForwardKernelData {α σ : Type} [Inhabited α] (obs : Observations α σ)
    {M : FP32 α} {rt : Runtime α σ} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    (e : ForwardData M rt o logits labels initial returned final) where
  launch : ForwardLaunchData M rt e.workingLogits.device e.parameters e.count e.zAllocated e.computed
  labelsSnapshot : SnapshotValues obs labels e.labels initial e.labelsReady
  working : Materialized obs logits e.workingLogits initial e.resolved
  layout : RowLayout e.workingLogits
  notNeg : e.workingLogits.isNeg=Bool.false
  lossAllocation : Allocation obs e.losses (if e.count>1 then [e.count,e.rows] else [e.rows])
    "torch.float32" e.workingLogits.device e.resolved e.lossAllocated
  lseAllocation : Allocation obs e.lse (if e.count>1 then [e.count,e.rows] else [e.rows])
    "torch.float32" e.workingLogits.device e.lossAllocated e.lseAllocated
  zAllocation : Allocation obs e.zlosses (if e.count>1 then [e.count,e.rows] else [e.rows])
    "torch.float32" e.workingLogits.device e.lseAllocated e.zAllocated
  lossDense : e.losses.Dense
  lseDense : e.lse.Dense
  zDense : e.zlosses.Dense
  labelsFrame : Preserves obs e.labelsReady launch.ready
  workingFrame : Preserves obs e.resolved launch.ready
  initialFrame : Preserves obs initial launch.ready
  lossLive : launch.ready.live e.losses.region
  lseLive : launch.ready.live e.lse.region
  zLive : launch.ready.live e.zlosses.region
  labelsLive : launch.ready.live e.labels.region
  workingLive : launch.ready.live e.workingLogits.region
  separation : Grid.ForwardSeparation e.parameters

theorem forward_kernel_data {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : LocalAPI obs rt) (groups : Groups obs rt.toRuntime) {M : FP32 α} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    (e : ForwardData M rt o logits labels initial returned final)
    (logitsLive : initial.live logits.region) (labelsLive : initial.live labels.region)
    (rank : logits.strides.length=logits.shape.length) : Nonempty (ForwardKernelData obs e) := by
  have prep := forward_preparation obs rt lib groups e logitsLive labelsLive rank
  have loss := lib.tensor.empty _ _ _ _ _ _ e.lossCall
  have lse := lib.tensor.empty _ _ _ _ _ _ e.lseCall
  have z := lib.tensor.empty _ _ _ _ _ _ e.zCall
  obtain ⟨launch⟩ := forward_launch_success M rt e.workingLogits.device e.parameters e.count e.zAllocated e.computed e.launchCall
  have enter := lib.tensor.enter _ _ _ launch.enterCall
  have setup := lib.setupForward _ _ _ _ launch.setupCall
  have afterZ := enter.trans setup
  have afterLse := z.1.frame.trans afterZ
  have afterLoss := lse.1.frame.trans afterLse
  have workingFrame := loss.1.frame.trans afterLoss
  have labelsFrame := prep.2.2.2.2.trans workingFrame
  have labelsResolved := prep.2.2.2.2.live _ prep.1.materialized.live
  have separation : Grid.ForwardSeparation e.parameters := by
    refine {outputDistinct:=?_,lseLogits:=?_,lossLogits:=?_,lseLabels:=?_,lossLabels:=?_,zLoss:=?_,zLse:=?_,zLogits:=?_,zLabels:=?_}
    · exact fresh_ne_live obs _ _ _ _ _ _ lse.1 _ loss.1.live
    · exact fresh_ne_live obs _ _ _ _ _ _ lse.1 _ (loss.1.frame.live _ prep.2.1.live)
    · exact fresh_ne_live obs _ _ _ _ _ _ loss.1 _ prep.2.1.live
    · simpa only [ForwardData.parameters,forwardParameters,CrossEntropy1.Host.forwardParameters,Region.cast_cast] using
        fresh_ne_live obs _ _ _ _ _ _ lse.1 _ (loss.1.frame.live _ labelsResolved)
    · simpa only [ForwardData.parameters,forwardParameters,CrossEntropy1.Host.forwardParameters,Region.cast_cast] using
        fresh_ne_live obs _ _ _ _ _ _ loss.1 _ labelsResolved
    · exact fresh_ne_live obs _ _ _ _ _ _ z.1 _ (lse.1.frame.live _ loss.1.live)
    · exact fresh_ne_live obs _ _ _ _ _ _ z.1 _ lse.1.live
    · exact fresh_ne_live obs _ _ _ _ _ _ z.1 _ (lse.1.frame.live _ (loss.1.frame.live _ prep.2.1.live))
    · simpa only [ForwardData.parameters,forwardParameters,CrossEntropy1.Host.forwardParameters,Region.cast_cast] using
        fresh_ne_live obs _ _ _ _ _ _ z.1 _ (lse.1.frame.live _ (loss.1.frame.live _ labelsResolved))
  exact ⟨⟨launch,prep.1,prep.2.1,prep.2.2.1,prep.2.2.2.1,loss.1,lse.1,z.1,loss.2,lse.2,z.2,
    labelsFrame,workingFrame,prep.2.1.frame.trans workingFrame,afterLoss.live _ loss.1.live,
    afterLse.live _ lse.1.live,afterZ.live _ z.1.live,labelsFrame.live _ prep.1.materialized.live,
    workingFrame.live _ prep.2.1.live,separation⟩⟩

theorem forward_kernel_logits {α σ : Type} [Inhabited α] (obs : Observations α σ)
    {M : FP32 α} {rt : Runtime α σ} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (dtype : logits.dtype="torch.float32") (row col : Nat) (hr : row<e.rows) (hc : col<e.cols) :
    (k.launch.ready.memory e.parameters.logits (CrossEntropy1.Source.logitAddress e.parameters.toParameters row col)).read .real=
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
    k.layout _ _ shape row col hr hc,ForwardData.parameters,forwardParameters,
    CrossEntropy1.Host.forwardParameters,CrossEntropy1.Source.logitAddress] using logical

theorem forward_kernel_labels {α σ : Type} [Inhabited α] (obs : Observations α σ)
    {M : FP32 α} {rt : Runtime α σ} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (dtype : labels.dtype="torch.int64") (row : Nat) (hr : row<e.rows) :
    CrossEntropy1.Source.label e.parameters.toParameters k.launch.ready.memory row=obs.integer initial labels row := by
  have inputSize : labels.numel=e.rows := by simp [Tensor.numel,e.labelsShape]
  have outputSize : e.labels.numel=e.rows := k.labelsSnapshot.materialized.metadata.numel.trans inputSize
  have base := dense_base e.labels k.labelsSnapshot.dense (by rw [outputSize]; omega)
  have cell := k.labelsSnapshot.labels dtype row (by simpa [inputSize] using hr)
  have frame := k.labelsFrame.memory e.labels.region row k.labelsSnapshot.materialized.live
  simp only [CrossEntropy1.Source.label,ForwardData.parameters,forwardParameters,CrossEntropy1.Host.forwardParameters,
    Region.cast_cast,Region.cast_id,base,Nat.zero_add,frame,cell,Cell.read_same]

theorem forward_kernel_outputs {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : LocalAPI obs rt) {M : FP32 α} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (row block : Nat) (hr : row<e.rows) (hb : block<e.count) :
    e.computed.memory e.parameters.loss (e.parameters.lossBase+block*e.rows+row)=
      .mk .real (Source.forwardLoss M e.parameters k.launch.ready.memory row block
        (CrossEntropy1.Source.label e.parameters.toParameters k.launch.ready.memory row)
        (Source.forwardValues M e.parameters k.launch.ready.memory row block)) ∧
    e.computed.memory e.parameters.lse (e.parameters.lseBase+block*e.rows+row)=
      .mk .real (Source.forwardValues M e.parameters k.launch.ready.memory row block).lse ∧
    (e.parameters.split=Bool.false → e.computed.memory e.parameters.zloss (e.parameters.zlossBase+block*e.rows+row)=
      .mk .real (Source.zLoss M e.parameters
        (CrossEntropy1.Source.label e.parameters.toParameters k.launch.ready.memory row)
        (Source.forwardValues M e.parameters k.launch.ready.memory row block).lse)) := by
  have exit := lib.tensor.exit _ _ _ k.launch.exitCall
  have results := Grid.forward_outputs M e.parameters k.launch.ready.memory e.count (row,block)
    ((CrossEntropy1.Grid.task_mem _ _ _).mpr ⟨hr,hb⟩) k.separation
  refine ⟨?_,?_,?_⟩
  · exact (exit.memory e.parameters.loss _ k.lossLive).trans results.1
  · exact (exit.memory e.parameters.lse _ k.lseLive).trans results.2.1
  · intro unsplit
    exact (exit.memory e.parameters.zloss _ k.zLive).trans (results.2.2 unsplit)

#axiomsClean group_metadata
#axiomsClean forward_preparation
#axiomsClean forward_kernel_data
#axiomsClean forward_kernel_logits
#axiomsClean forward_kernel_labels
#axiomsClean forward_kernel_outputs
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Host
