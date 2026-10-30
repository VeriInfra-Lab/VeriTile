import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.PeerRuns
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.ReturnedBackward
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.SavedBoundary

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Host
open CrossEntropy1.Host (World Observations IntegerViews CollectiveLibrary ObjectRegistry ViewGeometry
  RankInputs FloatArray value)
open CrossEntropy1.RealHost (arithmetic)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

/-- Actual split/distributed forward-to-autograd-backward composition. The
saved global statistic is the one derived from every peer's actual source
execution. No saved-LSE or expected-peer-output premise is introduced. -/
theorem split_forward_backward {σ : Type} (obs : Observations (WithBot ℝ) σ) (runs : Nat → RankRun σ obs)
    (world rows cols : Nat) (o : Options) (x : Nat → Nat → ℝ) (targets : Nat → Int)
    (group : CoherentRanks obs runs world rows cols o x targets) (rank : Nat) (rankBound : rank<world)
    (ints : IntegerViews obs) (gs gr : Nat → Nat)
    (comms : CollectiveLibrary obs arithmetic (runs rank).runtime.toRuntime gs gr)
    (size : (runs rank).forward.world>1 → (runs rank).forward.world=gs ((runs rank).options.processGroup.getD 0))
    (binding : RankInputs comms (runs rank).aggregate.rankCalls (peerLoss runs rows) (peerLSE runs rows))
    (objects : ObjectRegistry (runs rank).runtime.toRuntime) (views : ViewGeometry (runs rank).runtime.toRuntime)
    (logitsLive : (runs rank).initial.live (runs rank).logits.region)
    (strides : (runs rank).logits.strides.length=(runs rank).logits.shape.length)
    (registered : objects.valid (runs rank).logits)
    (versions : σ → Nat → Nat) (upstream output : Tensor) (zUpstream : Option Tensor) (backwardInitial final : World (WithBot ℝ) σ)
    (between : CrossEntropy1.Host.SavedPreserves obs (runs rank).returned.context.base (runs rank).final backwardInitial)
    (upstreamLive : backwardInitial.live upstream.region) (upstreamShape : upstream.shape=[rows])
    (upstreamDtype : upstream.dtype="torch.float32") (g : Nat → ℝ)
    (upstreams : ∀ row, row<rows → value obs backwardInitial upstream row=↑(g row))
    (backwardCall : backwardEntry RealSource.algebra (runs rank).runtime versions (runs rank).returned.context
      (runs rank).final.native upstream zUpstream backwardInitial=.ok output final) :
    output.shape=[rows,cols] ∧
    ((runs rank).options.inplace=Bool.true → output=(runs rank).returned.context.inplaceLogits) ∧
    ∀ row, row<rows →
      value obs (runs rank).final (runs rank).returned.losses row=
        ↑(Spec.loss (world*cols) o.logitScale o.smoothing o.squareScale (x row) (targets row) o.ignored) ∧
      value obs (runs rank).final (runs rank).returned.zLosses row=
        ↑(Spec.zLoss (world*cols) o.logitScale o.squareScale (x row) (targets row) o.ignored) ∧
      ∀ col, col<cols → value obs final output (row*cols+col)=
        ↑(Spec.gradient (world*cols) o.logitScale o.smoothing o.squareScale (g row) (x row)
          (targets row) o.ignored (rank*cols+col)) := by
  let run := runs rank
  have v := forward_view_data obs run.runtime run.library objects views run.kernel logitsLive strides registered
  have saved := split_saved obs arithmetic run.runtime run.library run.operations run.zeros ints objects gs gr comms
    run.kernel v (group.labelsDtype rank rankBound) size run.aggregate (peerLoss runs rows) (peerLSE runs rows) binding
  have fv := returned_split obs runs world rows cols o x targets group rank rankBound ints gs gr comms size binding
  have values := returned_split_values obs arithmetic run.runtime run.library run.operations run.zeros ints gs gr comms
    run.kernel (group.labelsDtype rank rankBound) size run.aggregate (peerLoss runs rows) (peerLSE runs rows) binding
  have options : run.returned.context.options=run.options := by simp only [run.forward.returnIdentity]
  have total : run.returned.context.totalClasses=world*cols := by
    simp only [run.forward.returnIdentity,run,group.worldEq rank rankBound,group.colsEq rank rankBound]
  have start : run.returned.context.classStart=rank*cols := by
    simp only [run.forward.returnIdentity,run,group.rankEq rank rankBound,group.colsEq rank rankBound]
  have logitsShape : run.returned.context.logits.shape=[rows,cols] := by
    simpa only [run,group.rowsEq rank rankBound,group.colsEq rank rankBound] using
      saved.logits.metadata.shape.trans run.forward.logitsShape
  have backwardSuccess := (backward_entry_success RealSource.algebra run.runtime versions run.returned.context run.final.native
    upstream output zUpstream backwardInitial final backwardCall).2
  obtain ⟨b⟩ := backward_success RealSource.algebra run.runtime run.returned.context upstream output zUpstream backwardInitial final backwardSuccess
  have dims : b.rows=rows ∧ b.cols=cols := by
    have h := b.shape.symm.trans logitsShape
    simpa only [List.cons.injEq,and_true] using h
  obtain ⟨bk⟩ := backward_kernel_data obs run.runtime run.library objects b (between.context saved.context) upstreamLive
  have lse : FloatArray obs run.final run.returned.context.lse [b.rows]
      (fun row => ↑(Spec.lse run.returned.context.totalClasses run.returned.context.options.logitScale (x row))) := by
    refine ⟨by simpa only [run,group.rowsEq rank rankBound,dims.1] using saved.lseShape,
      saved.lseDtype,saved.context.lseLive,?_⟩
    intro row hr
    have bound : row<rows := by
      simpa only [Tensor.numel,saved.lseShape,run,group.rowsEq rank rankBound,List.prod_cons,List.prod_nil,Nat.mul_one] using hr
    simpa only [total,options,run,group.logitScaleEq rank rankBound] using (fv row bound).2.1
  have savedLogits : ∀ row, row<b.rows → ∀ col, col<b.cols →
      value obs backwardInitial run.returned.context.logits (row*b.cols+col)=↑(x row (rank*cols+col)) := by
    intro row hr col hc
    have inputShape : run.logits.shape=[rows,cols] := by
      simpa only [run,group.rowsEq rank rankBound,group.colsEq rank rankBound] using run.forward.logitsShape
    have bound : row*b.cols+col<run.logits.numel := by
      simp only [Tensor.numel,inputShape,List.prod_cons,List.prod_nil,Nat.mul_one]
      have h := Nat.mul_le_mul_right cols (Nat.succ_le_of_lt (show row<rows by omega))
      simp only [Nat.succ_mul] at h
      rw [dims.2]
      omega
    exact (between.floatValue run.returned.context.logits (Or.inl rfl) _).trans
      ((saved.logits.floats (group.logitsDtype rank rankBound) _ bound).trans
        (by simpa only [dims.2] using group.inputs rank rankBound row (by omega) col (by omega)))
  have labels : CrossEntropy1.Host.IntArray obs run.final run.returned.context.labels [b.rows] targets := by
    refine ⟨by simpa only [run,group.rowsEq rank rankBound,dims.1] using saved.labels.shapeEq,
      saved.labels.dtype,saved.labels.live,?_⟩
    intro row hr
    have bound : row<rows := by
      simpa only [Tensor.numel,saved.labels.shapeEq,run,group.rowsEq rank rankBound,List.prod_cons,List.prod_nil,Nat.mul_one] using hr
    exact (saved.labels.values row hr).trans (group.labels rank rankBound row bound)
  have gradients := backward_real_values obs run.runtime run.library ints objects bk (between.context saved.context) saved.layout
    (saved.logits.metadata.dtype.trans (group.logitsDtype rank rankBound)) saved.lseDense saved.labelsDense (rank*cols) start
    (by simpa only [options,run,group.smoothingEq rank rankBound] using group.smoothingNonneg)
    x targets g savedLogits (between.lse lse) (between.labels labels) (by simpa only [dims.1] using upstreamShape) upstreamDtype
    (by simpa only [dims.1] using upstreams)
  refine ⟨bk.outputShape.trans logitsShape,?_,?_⟩
  · intro inplace
    simpa only [options,run,inplace,↓reduceIte] using b.outputIdentity
  · intro row hr
    refine ⟨(fv row hr).1,(fv row hr).2.2,?_⟩
    intro col hc
    simpa only [dims.2,total,options,run,group.logitScaleEq rank rankBound,group.smoothingEq rank rankBound,
      group.scaleEq rank rankBound,group.ignoredEq rank rankBound] using gradients row (by omega) col (by omega)

#axiomsClean split_forward_backward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Host
