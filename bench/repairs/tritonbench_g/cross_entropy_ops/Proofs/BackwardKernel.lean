import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.BackwardPreparation

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Host
open VeriTile.Triton
open CrossEntropy1.Source (FP32)
open CrossEntropy1.Host (World Step Observations ObjectRegistry SnapshotValues BufferInfo WorkInfo
  BackwardContext Preserves bwdWidth warps rowStride value snapshot_values flattened_slot)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

def BackwardData.parameters {α σ : Type} [Inhabited α] {M : FP32 α} {rt : Runtime α σ} {c : Context}
    {callerUpstream output : Tensor} {initial final : World α σ}
    (e : BackwardData M rt c callerUpstream output initial final) : Parameters :=
  backwardParameters c e.work e.upstream e.rows e.cols (bwdWidth e.cols)

def BackwardData.count {α σ : Type} [Inhabited α] {M : FP32 α} {rt : Runtime α σ} {c : Context}
    {callerUpstream output : Tensor} {initial final : World α σ}
    (e : BackwardData M rt c callerUpstream output initial final) : Nat := CrossEntropy1.Math.blocks e.cols (bwdWidth e.cols)

structure BackwardKernelData {α σ : Type} [Inhabited α] (obs : Observations α σ) {rt : Runtime α σ}
    (objects : ObjectRegistry rt.toRuntime) {M : FP32 α} {c : Context} {callerUpstream output : Tensor}
    {initial final : World α σ} (e : BackwardData M rt c callerUpstream output initial final) where
  snapshot : SnapshotValues obs callerUpstream e.upstream initial e.upstreamReady
  first : BufferInfo obs rt.toRuntime objects c.base e.upstream e.rows e.cols e.upstreamReady e.allocated e.first
  work : WorkInfo obs rt.toRuntime objects c.base e.upstream e.rows e.cols e.allocated e.prepared e.work
  launch : BackwardLaunchData M rt c.logits.device e.parameters e.count e.prepared e.computed
  initialFrame : Preserves obs initial launch.ready
  snapshotFrame : Preserves obs e.upstreamReady launch.ready
  workLive : launch.ready.live e.work.region
  outputLive : launch.ready.live output.region
  outputShape : output.shape=c.logits.shape
  outputObject : objects.valid output
  separation : Grid.BackwardSeparation e.parameters

theorem backward_kernel_data {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : LocalAPI obs rt) (objects : ObjectRegistry rt.toRuntime) {M : FP32 α} {c : Context}
    {callerUpstream output : Tensor} {initial final : World α σ}
    (e : BackwardData M rt c callerUpstream output initial final) (ctx : BackwardContext rt.toRuntime objects c.base initial)
    (upstreamLive : initial.live callerUpstream.region) : Nonempty (BackwardKernelData obs objects e) := by
  have snapshot := snapshot_values obs rt.toRuntime lib.tensor callerUpstream e.upstream initial e.upstreamReady upstreamLive e.snapshotCall
  have first := first_buffer obs rt lib objects e ctx snapshot
  have work := work_buffer obs rt lib objects e ctx snapshot first
  obtain ⟨launch⟩ := backward_launch_success M rt c.logits.device e.parameters e.count e.prepared e.computed e.launchCall
  have setupFrame := (lib.tensor.enter _ _ _ launch.enterCall).trans (lib.setupBackward _ _ _ _ launch.setupCall)
  have afterFirst := work.buffer.frame.trans setupFrame
  have afterSnapshot := first.frame.trans afterFirst
  have initialFrame := snapshot.materialized.frame.trans afterSnapshot
  have outputFacts : launch.ready.live output.region ∧ output.shape=c.logits.shape ∧ objects.valid output := by
    have identity := e.outputIdentity
    by_cases inplace : c.options.inplace=Bool.true
    · simp only [if_pos inplace] at identity
      exact ⟨by simpa only [identity] using initialFrame.live _ ctx.destinationLive,
        by simpa only [identity] using ctx.destinationShape,by simpa only [identity] using ctx.destinationObject⟩
    · simp only [if_neg inplace] at identity
      exact ⟨by simpa only [identity] using afterFirst.live _ first.live,
        by simpa only [identity] using first.shape,by simpa only [identity] using first.registered⟩
  have separation : Grid.BackwardSeparation e.parameters := by
    refine ⟨?_,work.buffer.upstream,work.buffer.lse,work.rows,?_⟩
    · simpa only [BackwardData.parameters,backwardParameters,CrossEntropy1.Host.backwardParameters,CrossEntropy1.Host.forwardParameters,Context.base,Region.cast_cast,Region.cast_id] using work.buffer.labels
    · intro sharedRegion
      have same : e.work=c.logits := work.buffer.logitsAlias sharedRegion
      exact ⟨congrArg Tensor.storageOffset same,congrArg rowStride same⟩
  exact ⟨⟨snapshot,first,work,launch,initialFrame,afterSnapshot,setupFrame.live _ work.buffer.live,
    outputFacts.1,outputFacts.2.1,outputFacts.2.2,separation⟩⟩

def BackwardKernelData.gradientValues {α σ : Type} [Inhabited α] {obs : Observations α σ} {rt : Runtime α σ}
    {objects : ObjectRegistry rt.toRuntime} {M : FP32 α} {c : Context} {callerUpstream output : Tensor}
    {initial final : World α σ} {e : BackwardData M rt c callerUpstream output initial final}
    (k : BackwardKernelData obs objects e) (i : Nat) : α :=
  Source.backwardValues M e.parameters k.launch.ready.memory (i/e.cols) ((i%e.cols)/e.parameters.block)
    ⟨(i%e.cols)%e.parameters.block,Nat.mod_lt _ (Nat.pos_of_ne_zero e.nonzero)⟩

/-- The actual backward grid writes every logical work coordinate exactly
once, even when the source input rows overlap in the out-of-place branch. -/
theorem backward_work_values {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : LocalAPI obs rt) {objects : ObjectRegistry rt.toRuntime} {M : FP32 α} {c : Context}
    {callerUpstream output : Tensor} {initial final : World α σ}
    {e : BackwardData M rt c callerUpstream output initial final} (k : BackwardKernelData obs objects e)
    (dtype : c.logits.dtype="torch.float32") :
    ∀ i, i<e.work.numel → value obs e.computed e.work i=k.gradientValues i := by
  intro i hi
  have shape := k.work.buffer.shape.trans e.shape
  have size : e.work.numel=e.rows*e.cols := by simp [Tensor.numel,shape]
  have slot := flattened_slot e.rows e.cols i (by simpa [size] using hi)
  have positive : 0<e.parameters.block := Nat.pos_of_ne_zero e.nonzero
  let col := i%e.cols
  let block := col/e.parameters.block
  let lane : Fin e.parameters.block := ⟨col%e.parameters.block,Nat.mod_lt col positive⟩
  have columnEq : CrossEntropy1.Source.column e.parameters.toParameters block lane.val=col := by
    simpa only [CrossEntropy1.Source.column,block,lane,Nat.mul_comm e.parameters.block] using Nat.div_add_mod col e.parameters.block
  have coverage := CrossEntropy1.Math.blocks_cover e.cols e.parameters.block positive
  have blockBound : block<e.count := by
    apply (Nat.div_lt_iff_lt_mul positive).mpr
    exact lt_of_lt_of_le slot.1 coverage
  have result := Grid.backward_output M e.parameters k.launch.ready.memory e.count (i/e.cols,block)
    ((CrossEntropy1.Grid.task_mem _ _ _).mpr ⟨slot.2.1,blockBound⟩) k.separation lane (by simpa only [columnEq] using slot.1)
  have exit := lib.tensor.exit _ _ _ k.launch.exitCall
  have stored := (exit.memory e.parameters.gradient (CrossEntropy1.Source.gradientAddress e.parameters.toParameters (i/e.cols) col) k.workLive).trans
    (by simpa only [columnEq] using result)
  have coordinate := k.work.layout e.rows e.cols shape (i/e.cols) col slot.2.1 slot.1
  rw [slot.2.2] at coordinate
  simp only [value,k.work.buffer.notNeg,Bool.false_eq_true,↓reduceIte,coordinate]
  change (e.computed.memory e.parameters.gradient (CrossEntropy1.Source.gradientAddress e.parameters.toParameters (i/e.cols) col)).read .real=_
  rw [stored,CrossEntropy1.Source.Cell.read_same]
  rfl

/-- Return the first allocated object, or the original saved in-place view.
When a work tensor is distinct, the copy contract transfers logical values,
including the sign-bit behavior of a lazy-negative destination view. -/
theorem backward_return_values {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : LocalAPI obs rt) {objects : ObjectRegistry rt.toRuntime} {M : FP32 α} {c : Context}
    {callerUpstream output : Tensor} {initial final : World α σ}
    {e : BackwardData M rt c callerUpstream output initial final} (k : BackwardKernelData obs objects e)
    (dtype : c.logits.dtype="torch.float32") :
    output.shape=c.logits.shape ∧ ∀ i, i<output.numel → value obs final output i=k.gradientValues i := by
  refine ⟨k.outputShape,?_⟩
  have values := backward_work_values obs rt lib k dtype
  rcases e.finish with ⟨same,state⟩|⟨different,copied,call⟩
  · have identity : e.work=output := objects.identity e.work output k.work.buffer.registered k.outputObject same
    intro i hi
    have bound : i<e.work.numel := by simpa only [identity] using hi
    have result := values i bound
    simpa only [state,identity] using result
  · have exit := lib.tensor.exit _ _ _ k.launch.exitCall
    have shape : output.shape=e.work.shape := k.outputShape.trans k.work.buffer.shape.symm
    have copy := lib.tensor.copy output e.work e.computed copied final (exit.live _ k.outputLive) (exit.live _ k.workLive) shape call
    intro i hi
    have bound : i<e.work.numel := by simpa only [Tensor.numel,← shape] using hi
    exact (copy.2.2.1 i hi).trans (values i bound)

#axiomsClean backward_kernel_data
#axiomsClean backward_work_values
#axiomsClean backward_return_values
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Host
