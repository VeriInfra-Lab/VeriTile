import bench.repairs.tritonbench_g.cross_entropy1.Proofs.SavedBoundary
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.RealBackward
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.PeerRuns

namespace VeriTile.Bench.Repairs.CrossEntropy1.RealHost
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

theorem context_preserve {α σ : Type} (obs : Observations α σ) {rt : Runtime α σ}
    {objects : ObjectRegistry rt} {c : Context} {a b : World α σ}
    (ctx : BackwardContext rt objects c a) (frame : Preserves obs a b) : BackwardContext rt objects c b :=
  ⟨frame.live _ ctx.logitsLive,frame.live _ ctx.labelsLive,frame.live _ ctx.lseLive,frame.live _ ctx.destinationLive,
    ctx.logitsRank,ctx.logitsNotNeg,ctx.logitsObject,ctx.destinationObject,ctx.destinationShape,
    ctx.logitsLabels,ctx.logitsLSE,ctx.rows⟩

private theorem backward_from_proved_lse {σ : Type} (obs : Observations (WithBot ℝ) σ) (rt : Runtime (WithBot ℝ) σ)
    (lib : Library obs rt) (ops : TensorLibrary obs arithmetic rt) (ints : IntegerViews obs) (objects : ObjectRegistry rt)
    (gs gr : Nat → Nat) (comms : CollectiveLibrary obs arithmetic rt gs gr)
    {o : Options} {logits labels : Tensor} {initial saved : World (WithBot ℝ) σ} {returned : ForwardReturn}
    {f : ForwardData RealSource.algebra rt o logits labels initial returned saved}
    (k : ForwardKernelData obs f) (v : ForwardViewData obs objects f)
    (dtype : logits.dtype="torch.float32") (labelDtype : labels.dtype="torch.int64")
    (size : f.world>1 → f.world=gs (o.processGroup.getD 0)) (aggregation : ForwardAggregation f)
    (lossPeers lsePeers : Nat → WithBot ℝ) (inputs : ForwardInputs comms aggregation lossPeers lsePeers)
    (smooth : 0≤o.smoothing) (x : Nat → Nat → ℝ)
    (caller : ∀ row, row<f.rows → ∀ col, col<f.cols → value obs initial logits (row*f.cols+col)=↑(x row (f.rank*f.cols+col)))
    (forwardLSE : ∀ row, row<f.rows → value obs saved returned.context.lse row=↑(Math.lse (f.world*f.cols) (x row)))
    {before after : World (WithBot ℝ) σ} (between : SavedPreserves obs returned.context saved before)
    (callerUpstream output : Tensor) (upstreamLive : before.live callerUpstream.region)
    (upstreamShape : callerUpstream.shape=[f.rows]) (upstreamDtype : callerUpstream.dtype="torch.float32")
    (upstream : Nat → ℝ) (upstreamValues : ∀ row, row<f.rows → value obs before callerUpstream row=↑(upstream row))
    (b : BackwardData RealSource.algebra rt returned.context callerUpstream output before after) :
    ∀ row, row<f.rows → ∀ col, col<f.cols → value obs after output (row*f.cols+col)=
      ↑(Math.gradient (f.world*f.cols) (x row) (obs.integer initial labels row) o.ignored o.smoothing o.squareScale
        (upstream row) (f.rank*f.cols+col)) := by
  have forwardValues := forward_values obs arithmetic rt lib ops ints gs gr comms k labelDtype size aggregation lossPeers lsePeers inputs
  have ctxSaved := forward_backward_context obs arithmetic rt lib ops ints objects gs gr comms k v labelDtype size aggregation lossPeers lsePeers inputs
  have ctx := between.context ctxSaved
  obtain ⟨bk⟩ := backward_kernel_data obs rt lib objects b ctx upstreamLive
  have materialized := forwardValues.2.2.2.1
  have logitsEq : returned.context.logits=f.workingLogits := congrArg (fun r : ForwardReturn => r.context.logits) f.returnIdentity
  have labelsEq : returned.context.labels=f.labels := congrArg (fun r : ForwardReturn => r.context.labels) f.returnIdentity
  have lseEq : returned.context.lse=f.pair.2 := congrArg (fun r : ForwardReturn => r.context.lse) f.returnIdentity
  have optionsEq : returned.context.options=o := congrArg (fun r : ForwardReturn => r.context.options) f.returnIdentity
  have totalEq : returned.context.totalClasses=f.world*f.cols := congrArg (fun r : ForwardReturn => r.context.totalClasses) f.returnIdentity
  have startEq : returned.context.classStart=((f.rank*f.cols : Nat) : Int) :=
    congrArg (fun r : ForwardReturn => (r.context.classStart : Int)) f.returnIdentity
  have forwardShape : returned.context.logits.shape=[f.rows,f.cols] :=
    materialized.metadata.shape.trans f.logitsShape
  have shapeEq : [b.rows,b.cols]=[f.rows,f.cols] := b.shape.symm.trans forwardShape
  have dimensions : b.rows=f.rows ∧ b.cols=f.cols := by simpa only [List.cons.injEq,and_true] using shapeEq
  have savedLogits : ∀ row, row<b.rows → ∀ col, col<b.cols →
      value obs before returned.context.logits (row*b.cols+col)=↑(x row (f.rank*f.cols+col)) := by
    intro row hr col hc
    have bound : row*b.cols+col<logits.numel := by
      simp only [Tensor.numel,f.logitsShape,List.prod_cons,List.prod_nil,Nat.mul_one,← dimensions.1,← dimensions.2]
      have product := Nat.mul_le_mul_right b.cols (Nat.succ_le_of_lt hr)
      simp only [Nat.succ_mul] at product
      omega
    rw [between.floatValue returned.context.logits (Or.inl rfl),materialized.floats dtype _ bound]
    simpa only [dimensions.2] using caller row (by simpa only [dimensions.1] using hr) col (by simpa only [dimensions.2] using hc)
  have lseBefore : FloatArray obs before returned.context.lse [b.rows]
      (fun row => ↑(Math.lse returned.context.totalClasses (x row))) := by
    refine ⟨by simpa only [dimensions.1] using forwardValues.2.1.shapeEq,forwardValues.2.1.dtype,
      between.live _ forwardValues.2.1.live,?_⟩
    intro row hr
    have bound : row<f.rows := by simpa [Tensor.numel,forwardValues.2.1.shapeEq] using hr
    rw [between.floatValue returned.context.lse (Or.inr (Or.inl rfl)) row,forwardLSE row bound,totalEq]
  have labelsBefore : IntArray obs before returned.context.labels [b.rows] (obs.integer initial labels) := by
    simpa only [dimensions.1] using between.labels forwardValues.2.2.1
  have lseDense := (forward_saved_lse obs arithmetic rt lib ops ints gs gr comms k size aggregation).1
  have result := backward_real_values obs rt lib ints objects bk ctx
    (by simpa only [logitsEq] using k.layout) (materialized.metadata.dtype.trans dtype)
    (by simpa only [lseEq] using lseDense) (by simpa only [labelsEq] using k.labelsSnapshot.dense)
    (f.rank*f.cols) startEq (by simpa only [optionsEq] using smooth) x (obs.integer initial labels) upstream
    savedLogits lseBefore labelsBefore (by simpa only [dimensions.1] using upstreamShape) upstreamDtype
    (by simpa only [dimensions.1] using upstreamValues)
  simpa only [dimensions.1,dimensions.2,totalEq,optionsEq] using result

/-- Whole unsplit Real execution. The saved LSE needed by backward is proved
from the actual preceding forward call inside this theorem. -/
theorem complete_unsplit {σ : Type} (obs : Observations (WithBot ℝ) σ) (rt : Runtime (WithBot ℝ) σ)
    (lib : Library obs rt) (ops : TensorLibrary obs arithmetic rt) (ints : IntegerViews obs) (objects : ObjectRegistry rt)
    (gs gr : Nat → Nat) (comms : CollectiveLibrary obs arithmetic rt gs gr)
    {o : Options} {logits labels : Tensor} {initial saved : World (WithBot ℝ) σ} {returned : ForwardReturn}
    {f : ForwardData RealSource.algebra rt o logits labels initial returned saved}
    (k : ForwardKernelData obs f) (v : ForwardViewData obs objects f)
    (dtype : logits.dtype="torch.float32") (labelDtype : labels.dtype="torch.int64")
    (worldPositive : 0<f.world) (rankBound : f.rank<f.world) (unsplit : ¬(f.world>1 ∨ f.cols>65536))
    (smooth : 0≤o.smoothing) (x : Nat → Nat → ℝ)
    (caller : ∀ row, row<f.rows → ∀ col, col<f.cols → value obs initial logits (row*f.cols+col)=↑(x row col))
    (valid : ∀ row, row<f.rows → obs.integer initial labels row=o.ignored ∨
      (0≤obs.integer initial labels row ∧ obs.integer initial labels row<(f.world*f.cols : Nat)))
    {before after : World (WithBot ℝ) σ} (between : SavedPreserves obs returned.context saved before)
    (callerUpstream output : Tensor) (upstreamLive : before.live callerUpstream.region)
    (upstreamShape : callerUpstream.shape=[f.rows]) (upstreamDtype : callerUpstream.dtype="torch.float32")
    (upstream : Nat → ℝ) (upstreamValues : ∀ row, row<f.rows → value obs before callerUpstream row=↑(upstream row))
    (b : BackwardData RealSource.algebra rt returned.context callerUpstream output before after) :
    (∀ row, row<f.rows → value obs saved returned.losses row=
      ↑(Math.loss (f.world*f.cols) (x row) (obs.integer initial labels row) o.ignored o.smoothing o.squareScale)) ∧
    (∀ row, row<f.rows → ∀ col, col<f.cols → value obs after output (row*f.cols+col)=
      ↑(Math.gradient (f.world*f.cols) (x row) (obs.integer initial labels row) o.ignored o.smoothing o.squareScale (upstream row) col)) := by
  have forward := returned_unsplit obs rt lib ops ints gs gr comms k dtype labelDtype worldPositive rankBound unsplit smooth x caller valid
  obtain ⟨aggregation⟩ := forward_aggregation f
  have inputs : ∀ lp sp, ForwardInputs comms aggregation lp sp := by
    cases aggregation with
    | direct => intro _ _; trivial
    | split flag => exact False.elim (unsplit flag)
  have rank : f.rank=0 := by omega
  let peers : Nat → WithBot ℝ := fun _ => ↑(0 : ℝ)
  have backward := backward_from_proved_lse obs rt lib ops ints objects gs gr comms k v dtype labelDtype
    (fun many => False.elim (unsplit (Or.inl many))) aggregation peers peers (inputs peers peers) smooth x
    (by simpa only [rank,Nat.zero_mul,Nat.zero_add] using caller)
    (fun row hr => (forward row hr).2) between callerUpstream output upstreamLive upstreamShape upstreamDtype upstream upstreamValues b
  exact ⟨fun row hr => (forward row hr).1,by simpa only [rank,Nat.zero_mul,Nat.zero_add] using backward⟩

/-- Whole split execution, including actual peer-local computation and
collective transmission, saved forward data, and the returned backward object. -/
theorem complete_split {σ : Type} (obs : Observations (WithBot ℝ) σ) (runs : Nat → RankRun σ obs)
    (world rows cols : Nat) (o : Options) (x : Nat → Nat → ℝ) (targets : Nat → Int)
    (group : CoherentRanks obs runs world rows cols o x targets) (rank : Nat) (rankBound : rank<world)
    (ints : IntegerViews obs) (gs gr : Nat → Nat)
    (comms : CollectiveLibrary obs arithmetic (runs rank).runtime gs gr)
    (size : (runs rank).forward.world>1 → (runs rank).forward.world=gs ((runs rank).options.processGroup.getD 0))
    (binding : RankInputs comms (runs rank).aggregate.rankCalls (peerLoss runs rows) (peerLSE runs rows))
    (objects : ObjectRegistry (runs rank).runtime) (views : ForwardViewData obs objects (runs rank).forward)
    {before after : World (WithBot ℝ) σ} (between : SavedPreserves obs (runs rank).returned.context (runs rank).final before)
    (callerUpstream output : Tensor) (upstreamLive : before.live callerUpstream.region)
    (upstreamShape : callerUpstream.shape=[rows]) (upstreamDtype : callerUpstream.dtype="torch.float32")
    (upstream : Nat → ℝ) (upstreamValues : ∀ row, row<rows → value obs before callerUpstream row=↑(upstream row))
    (b : BackwardData RealSource.algebra (runs rank).runtime (runs rank).returned.context callerUpstream output before after) :
    (∀ row, row<rows → value obs (runs rank).final (runs rank).returned.losses row=
      ↑(Math.loss (world*cols) (x row) (targets row) o.ignored o.smoothing o.squareScale)) ∧
    (∀ row, row<rows → ∀ col, col<cols → value obs after output (row*cols+col)=
      ↑(Math.gradient (world*cols) (x row) (targets row) o.ignored o.smoothing o.squareScale (upstream row) (rank*cols+col))) := by
  have forward := returned_split obs runs world rows cols o x targets group rank rankBound ints gs gr comms size binding
  have backward := backward_from_proved_lse obs (runs rank).runtime (runs rank).library (runs rank).operations
    ints objects gs gr comms (runs rank).kernel views (group.logitsDtype rank rankBound) (group.labelsDtype rank rankBound)
    size (.split (runs rank).split (runs rank).aggregate) (peerLoss runs rows) (peerLSE runs rows) binding
    (by simpa only [group.smoothingEq rank rankBound] using group.smoothingNonneg) x
    (by simpa only [group.rowsEq rank rankBound,group.colsEq rank rankBound,group.rankEq rank rankBound] using group.inputs rank rankBound)
    (by simpa only [group.rowsEq rank rankBound,group.worldEq rank rankBound,group.colsEq rank rankBound] using fun row hr => (forward row hr).2)
    between callerUpstream output upstreamLive (by simpa only [group.rowsEq rank rankBound] using upstreamShape) upstreamDtype
    upstream (by simpa only [group.rowsEq rank rankBound] using upstreamValues) b
  refine ⟨fun row hr => (forward row hr).1,?_⟩
  intro row hr col hc
  have result := backward row (by simpa only [group.rowsEq rank rankBound] using hr)
    col (by simpa only [group.colsEq rank rankBound] using hc)
  simpa only [group.colsEq rank rankBound,group.worldEq rank rankBound,group.rankEq rank rankBound,
    group.labels rank rankBound row hr,group.ignoredEq rank rankBound,group.smoothingEq rank rankBound,
    group.scaleEq rank rankBound] using result

#axiomsClean complete_unsplit
#axiomsClean complete_split
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.RealHost
