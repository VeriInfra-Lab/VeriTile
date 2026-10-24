import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.Preparation
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.Real
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.RealUnsplit

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Host
open VeriTile.Triton
open CrossEntropy1.Source (Cell)
open CrossEntropy1.Host (World Step Observations Preserves Groups value fwdWidth
  forward_width one_block dense_base)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

theorem caller_input {σ : Type} (obs : Observations (WithBot ℝ) σ)
    {rt : Runtime (WithBot ℝ) σ} {o : Options} {logits labels : Tensor}
    {initial final : World (WithBot ℝ) σ} {returned : ForwardReturn}
    {e : ForwardData RealSource.algebra rt o logits labels initial returned final}
    (k : ForwardKernelData obs e) (dtype : logits.dtype="torch.float32")
    (row : Nat) (hr : row<e.rows) (x : Nat → ℝ)
    (caller : ∀ col, col<e.cols → value obs initial logits (row*e.cols+col)=↑(x col)) :
    RealSource.Input e.parameters k.launch.ready.memory row x := by
  intro col hc
  exact (forward_kernel_logits obs k dtype row col hr hc).trans (caller col hc)

theorem unsplit_identity {α σ : Type} [Inhabited α] {M : CrossEntropy1.Source.FP32 α}
    {rt : Runtime α σ} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    (e : ForwardData M rt o logits labels initial returned final)
    (unsplit : ¬(e.world>1 ∨ e.cols>65536)) :
    e.result=⟨e.losses,e.lse,e.zlosses⟩ ∧ e.aggregated=e.computed ∧ e.count=1 := by
  have call := e.aggregateCall
  simp only [unsplit,decide_false,Bool.false_eq_true,↓reduceIte,Step.ok.injEq] at call
  have count := one_block e.cols (fwdWidth e.cols) (Nat.pos_of_ne_zero e.positive)
    ((forward_width e.cols (Nat.pos_of_ne_zero e.positive)).2.2 (by omega))
  exact ⟨call.1.symm,call.2.symm,count⟩

/-- Readback uses the actual dense allocation and actual saved/marked states.
No premise specifies a returned value or an already-correct saved statistic. -/
theorem returned_unsplit {σ : Type} (obs : Observations (WithBot ℝ) σ)
    (rt : Runtime (WithBot ℝ) σ) (lib : LocalAPI obs rt) (groups : Groups obs rt.toRuntime)
    {o : Options} {logits labels : Tensor} {initial final : World (WithBot ℝ) σ}
    {returned : ForwardReturn} {e : ForwardData RealSource.algebra rt o logits labels initial returned final}
    (k : ForwardKernelData obs e) (dtype : logits.dtype="torch.float32") (labelDtype : labels.dtype="torch.int64")
    (unsplit : ¬(e.world>1 ∨ e.cols>65536)) (smooth : 0≤o.smoothing)
    (x : Nat → Nat → ℝ)
    (caller : ∀ row, row<e.rows → ∀ col, col<e.cols → value obs initial logits (row*e.cols+col)=↑(x row col))
    (valid : ∀ row, row<e.rows → obs.integer initial labels row=o.ignored ∨
      (0≤obs.integer initial labels row ∧ obs.integer initial labels row<(e.world*e.cols : Nat))) :
    returned.losses.shape=[e.rows] ∧ returned.zLosses.shape=[e.rows] ∧ returned.context.lse.shape=[e.rows] ∧
    ∀ row, row<e.rows →
      value obs final returned.losses row=
        ↑(Spec.loss e.cols o.logitScale o.smoothing o.squareScale (x row) (obs.integer initial labels row) o.ignored) ∧
      value obs final returned.context.lse row=↑(Spec.lse e.cols o.logitScale (x row)) ∧
      value obs final returned.zLosses row=
        ↑(Spec.zLoss e.cols o.logitScale o.squareScale (x row) (obs.integer initial labels row) o.ignored) := by
  have ids := unsplit_identity e unsplit
  have metadata := group_metadata obs rt groups o.processGroup initial e.sized e.ranked e.world e.rank e.sizeCall e.rankCall
  have world : e.world=1 := by omega
  have rank : e.rank=0 := by omega
  have positive := Nat.pos_of_ne_zero e.positive
  have width := forward_width e.cols positive
  have covered := width.2.2 (show e.cols≤65536 by omega)
  have split : e.parameters.split=Bool.false := by
    simp [ForwardData.parameters,forwardParameters,CrossEntropy1.Host.forwardParameters,unsplit]
  have post : Preserves obs e.computed final := by
    rw [← ids.2.1]
    exact (lib.tensor.save _ _ _ _ _ _ e.saveCall).trans (lib.mark _ _ _ e.markCall)
  have exit := lib.tensor.exit _ _ _ k.launch.exitCall
  have lossShape : e.losses.shape=[e.rows] := by simpa [ids.2.2] using k.lossAllocation.shapeEq
  have lseShape : e.lse.shape=[e.rows] := by simpa [ids.2.2] using k.lseAllocation.shapeEq
  have zShape : e.zlosses.shape=[e.rows] := by simpa [ids.2.2] using k.zAllocation.shapeEq
  simp only [e.returnIdentity,ids.1]
  refine ⟨lossShape,zShape,lseShape,?_⟩
  intro row hr
  have lossSize : e.losses.numel=e.rows := by simp [Tensor.numel,lossShape]
  have lseSize : e.lse.numel=e.rows := by simp [Tensor.numel,lseShape]
  have zSize : e.zlosses.numel=e.rows := by simp [Tensor.numel,zShape]
  have lossBase := dense_base e.losses k.lossDense (by rw [lossSize]; omega)
  have lseBase := dense_base e.lse k.lseDense (by rw [lseSize]; omega)
  have zBase := dense_base e.zlosses k.zDense (by rw [zSize]; omega)
  have input := caller_input obs k dtype row hr (x row) (caller row hr)
  have label := forward_kernel_labels obs k labelDtype row hr
  have stores := forward_kernel_outputs obs rt lib k row 0 hr (by rw [ids.2.2]; omega)
  have saved := RealSource.forward_lse e.parameters k.launch.ready.memory row 0 (x row) input width.1
    (by simpa [ForwardData.parameters,forwardParameters] using positive)
  rw [CrossEntropy1.Math.block_zero_lse e.parameters.cols e.parameters.block covered] at saved
  have loss := RealSource.forward_unsplit_loss e.parameters k.launch.ready.memory row (x row) input
    (obs.integer initial labels row) positive covered split
    (CrossEntropy1.RealHost.smoothing_zero o.toOptions rfl rfl smooth)
    (by simp [ForwardData.parameters,forwardParameters,CrossEntropy1.Host.forwardParameters,rank])
    (by simp [ForwardData.parameters,forwardParameters,CrossEntropy1.Host.forwardParameters,world])
    (by simpa [ForwardData.parameters,forwardParameters,CrossEntropy1.Host.forwardParameters,world] using valid row hr)
  have rawLoss : value obs e.computed e.losses row=↑(Spec.loss e.cols o.logitScale o.smoothing o.squareScale
      (x row) (obs.integer initial labels row) o.ignored) := by
    rw [value,k.lossAllocation.notNeg]
    simp only [Bool.false_eq_true,↓reduceIte,k.lossDense.2 row (by simpa [lossSize] using hr)]
    have stored := stores.1
    change e.computed.memory e.losses.region (e.losses.storageOffset+0*e.rows+row)=_ at stored
    simp only [lossBase,Nat.zero_mul,Nat.zero_add] at stored
    rw [stored,Cell.read_same,label]
    exact loss
  have rawLSE : value obs e.computed e.lse row=↑(Spec.lse e.cols o.logitScale (x row)) := by
    rw [value,k.lseAllocation.notNeg]
    simp only [Bool.false_eq_true,↓reduceIte,k.lseDense.2 row (by simpa [lseSize] using hr)]
    have stored := stores.2.1
    change e.computed.memory e.lse.region (e.lse.storageOffset+0*e.rows+row)=_ at stored
    simp only [lseBase,Nat.zero_mul,Nat.zero_add] at stored
    rw [stored,Cell.read_same]
    exact saved
  have rawZ : value obs e.computed e.zlosses row=↑(Spec.zLoss e.cols o.logitScale o.squareScale
      (x row) (obs.integer initial labels row) o.ignored) := by
    rw [value,k.zAllocation.notNeg]
    simp only [Bool.false_eq_true,↓reduceIte,k.zDense.2 row (by simpa [zSize] using hr)]
    have stored := stores.2.2 split
    change e.computed.memory e.zlosses.region (e.zlosses.storageOffset+0*e.rows+row)=_ at stored
    simp only [zBase,Nat.zero_mul,Nat.zero_add] at stored
    rw [stored,Cell.read_same,label,saved,RealSource.logging_zloss _ _ _ split]
    rfl
  exact ⟨(post.floatValue _ (exit.live _ k.lossLive) row).trans rawLoss,
    (post.floatValue _ (exit.live _ k.lseLive) row).trans rawLSE,
    (post.floatValue _ (exit.live _ k.zLive) row).trans rawZ⟩

#axiomsClean caller_input
#axiomsClean unsplit_identity
#axiomsClean returned_unsplit
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Host
