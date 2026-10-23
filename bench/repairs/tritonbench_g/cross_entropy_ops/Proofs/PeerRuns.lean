import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.RealLocal
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.RealRanks

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Host
open CrossEntropy1.Host (World Observations TensorLibrary IntegerViews CollectiveLibrary
  RankInputs value localResult local_reduction_values rankResult)
open CrossEntropy1.RealHost (arithmetic block_coordinates)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

/-- One peer's actual wrapper and source-grid trace. Every peer carries its
own ordinary runtime contracts; process-local group ranks need not share a
runtime implementation or state. -/
structure RankRun (σ : Type) (obs : Observations (WithBot ℝ) σ) where
  runtime : Runtime (WithBot ℝ) σ
  library : LocalAPI obs runtime
  operations : TensorLibrary obs arithmetic runtime.toRuntime
  zeros : ZeroAPI obs runtime arithmetic.zero
  options : Options
  logits : Tensor
  labels : Tensor
  initial : World (WithBot ℝ) σ
  final : World (WithBot ℝ) σ
  returned : ForwardReturn
  forward : ForwardData RealSource.algebra runtime options logits labels initial returned final
  kernel : ForwardKernelData obs forward
  aggregate : AggregateData runtime options forward.world forward.rows forward.count forward.labels forward.losses forward.lse
    forward.computed forward.result forward.aggregated
  split : forward.world>1 ∨ forward.cols>65536

def RankRun.localLoss {σ : Type} {obs : Observations (WithBot ℝ) σ} (run : RankRun σ obs) (row : Nat) : WithBot ℝ :=
  value obs run.aggregate.localState run.aggregate.localPair.1 row

def RankRun.localLSE {σ : Type} {obs : Observations (WithBot ℝ) σ} (run : RankRun σ obs) (row : Nat) : WithBot ℝ :=
  value obs run.aggregate.localState run.aggregate.localPair.2 row

def peerLoss {σ : Type} {obs : Observations (WithBot ℝ) σ} (runs : Nat → RankRun σ obs) (rows i : Nat) : WithBot ℝ :=
  (runs (i/rows)).localLoss (i%rows)

def peerLSE {σ : Type} {obs : Observations (WithBot ℝ) σ} (runs : Nat → RankRun σ obs) (rows i : Nat) : WithBot ℝ :=
  (runs (i/rows)).localLSE (i%rows)

/-- Shared task input and group metadata. These are caller values and actual
rank-query results, not peer-output formulas. -/
structure CoherentRanks {σ : Type} (obs : Observations (WithBot ℝ) σ) (runs : Nat → RankRun σ obs)
    (world rows cols : Nat) (o : Options) (x : Nat → Nat → ℝ) (targets : Nat → Int) : Prop where
  worldPositive : 0<world
  colsPositive : 0<cols
  smoothingNonneg : 0≤o.smoothing
  worldEq : ∀ r, r<world → (runs r).forward.world=world
  rankEq : ∀ r, r<world → (runs r).forward.rank=r
  rowsEq : ∀ r, r<world → (runs r).forward.rows=rows
  colsEq : ∀ r, r<world → (runs r).forward.cols=cols
  smoothingEq : ∀ r, r<world → (runs r).options.smoothing=o.smoothing
  scaleEq : ∀ r, r<world → (runs r).options.squareScale=o.squareScale
  logitScaleEq : ∀ r, r<world → (runs r).options.logitScale=o.logitScale
  groupEq : ∀ r, r<world → (runs r).options.processGroup=o.processGroup
  ignoredEq : ∀ r, r<world → (runs r).options.ignored=o.ignored
  logitsDtype : ∀ r, r<world → (runs r).logits.dtype="torch.float32"
  labelsDtype : ∀ r, r<world → (runs r).labels.dtype="torch.int64"
  inputs : ∀ r, r<world → ∀ row, row<rows → ∀ col, col<cols →
    value obs (runs r).initial (runs r).logits (row*cols+col)=↑(x row (r*cols+col))
  labels : ∀ r, r<world → ∀ row, row<rows → obs.integer (runs r).initial (runs r).labels row=targets row
  valid : ∀ row, row<rows → targets row=o.ignored ∨ (0≤targets row ∧ targets row<(world*cols : Nat))

theorem actual_peer_values {σ : Type} (obs : Observations (WithBot ℝ) σ) (runs : Nat → RankRun σ obs)
    (world rows cols : Nat) (o : Options) (x : Nat → Nat → ℝ) (targets : Nat → Int)
    (group : CoherentRanks obs runs world rows cols o x targets) (rank row : Nat) (rankBound : rank<world) (rowBound : row<rows) :
    (runs rank).localLoss row=↑(if targets row=o.ignored then 0 else
      CrossEntropy1.Aggregation.rankContribution world cols (targets row).toNat o.smoothing (Spec.scaled o.logitScale (x row)) rank) ∧
    (runs rank).localLSE row=↑(Spec.lse cols o.logitScale (fun col => x row (rank*cols+col))) := by
  have hr : row<(runs rank).forward.rows := by simpa only [group.rowsEq rank rankBound] using rowBound
  have caller : ∀ col, col<(runs rank).forward.cols →
      value obs (runs rank).initial (runs rank).logits (row*(runs rank).forward.cols+col)=
        ↑(x row ((runs rank).forward.rank*(runs rank).forward.cols+col)) := by
    intro col hc
    simpa only [group.colsEq rank rankBound,group.rankEq rank rankBound] using
      group.inputs rank rankBound row rowBound col (by simpa only [group.colsEq rank rankBound] using hc)
  have valid : obs.integer (runs rank).initial (runs rank).labels row=(runs rank).options.ignored ∨
      (0≤obs.integer (runs rank).initial (runs rank).labels row ∧
        obs.integer (runs rank).initial (runs rank).labels row<((runs rank).forward.world*(runs rank).forward.cols : Nat)) := by
    simpa only [group.labels rank rankBound row rowBound,group.ignoredEq rank rankBound,
      group.worldEq rank rankBound,group.colsEq rank rankBound] using group.valid row rowBound
  have result := local_pair_values obs (runs rank).runtime (runs rank).library (runs rank).operations
    (runs rank).kernel (runs rank).aggregate (group.logitsDtype rank rankBound) (group.labelsDtype rank rankBound)
    (runs rank).split (by simpa only [group.smoothingEq rank rankBound] using group.smoothingNonneg)
    row hr (x row) caller valid
  simpa only [RankRun.localLoss,RankRun.localLSE,group.labels rank rankBound row rowBound,
    group.ignoredEq rank rankBound,group.worldEq rank rankBound,group.colsEq rank rankBound,
    group.rankEq rank rankBound,group.smoothingEq rank rankBound,group.logitScaleEq rank rankBound] using result

theorem transmitted_peer_values {σ : Type} (obs : Observations (WithBot ℝ) σ) (runs : Nat → RankRun σ obs)
    (world rows cols : Nat) (o : Options) (x : Nat → Nat → ℝ) (targets : Nat → Int)
    (group : CoherentRanks obs runs world rows cols o x targets) (rank row : Nat) (rankBound : rank<world) (rowBound : row<rows) :
    peerLoss runs rows (rank*rows+row)=↑(if targets row=o.ignored then 0 else
      CrossEntropy1.Aggregation.rankContribution world cols (targets row).toNat o.smoothing (Spec.scaled o.logitScale (x row)) rank) ∧
    peerLSE runs rows (rank*rows+row)=↑(Spec.lse cols o.logitScale (fun col => x row (rank*cols+col))) := by
  have coordinates := block_coordinates rows row rank rowBound
  simpa only [peerLoss,peerLSE,coordinates.1,coordinates.2] using
    actual_peer_values obs runs world rows cols o x targets group rank row rankBound rowBound

/-- All three actual return values of the chosen rank. Communication binds
the transmitted buffers to actual peer local outputs, whose formulas are
proved by `actual_peer_values`, rather than supplied as correctness premises. -/
theorem returned_split {σ : Type} (obs : Observations (WithBot ℝ) σ) (runs : Nat → RankRun σ obs)
    (world rows cols : Nat) (o : Options) (x : Nat → Nat → ℝ) (targets : Nat → Int)
    (group : CoherentRanks obs runs world rows cols o x targets) (rank : Nat) (rankBound : rank<world)
    (ints : IntegerViews obs) (gs gr : Nat → Nat)
    (comms : CollectiveLibrary obs arithmetic (runs rank).runtime.toRuntime gs gr)
    (size : (runs rank).forward.world>1 → (runs rank).forward.world=gs ((runs rank).options.processGroup.getD 0))
    (binding : RankInputs comms (runs rank).aggregate.rankCalls (peerLoss runs rows) (peerLSE runs rows)) :
    ∀ row, row<rows →
      value obs (runs rank).final (runs rank).returned.losses row=
        ↑(Spec.loss (world*cols) o.logitScale o.smoothing o.squareScale (x row) (targets row) o.ignored) ∧
      value obs (runs rank).final (runs rank).returned.context.lse row=↑(Spec.lse (world*cols) o.logitScale (x row)) ∧
      value obs (runs rank).final (runs rank).returned.zLosses row=
        ↑(Spec.zLoss (world*cols) o.logitScale o.squareScale (x row) (targets row) o.ignored) := by
  let run := runs rank
  have values := returned_split_values obs arithmetic run.runtime run.library run.operations run.zeros ints gs gr comms
    run.kernel (group.labelsDtype rank rankBound) size run.aggregate (peerLoss runs rows) (peerLSE runs rows) binding
  have arrays := forward_arrays obs run.runtime run.library run.kernel
  have locals := local_reduction_values obs arithmetic run.runtime.toRuntime run.operations run.forward.count run.forward.rows
    run.forward.losses run.forward.lse run.forward.computed run.aggregate.localPair run.aggregate.localState
    run.kernel.lossValues run.kernel.lseValues arrays.1 arrays.2 run.kernel.separation.outputDistinct.symm run.aggregate.localCalls
  intro row rowBound
  have hr : row<run.forward.rows := by simpa only [run,group.rowsEq rank rankBound] using rowBound
  have localLossBound : row<run.aggregate.localPair.1.numel := by simpa [Tensor.numel,locals.losses.shapeEq] using hr
  have localLSEBound : row<run.aggregate.localPair.2.numel := by simpa [Tensor.numel,locals.lse.shapeEq] using hr
  have peer := actual_peer_values obs runs world rows cols o x targets group rank row rankBound rowBound
  have math := rank_loss_math o world cols rows row rank group.worldPositive group.colsPositive rankBound (targets row) (x row)
    (group.valid row rowBound)
    (localResult arithmetic.sum run.forward.count run.forward.rows run.kernel.lossValues)
    (localResult arithmetic.logsumexp run.forward.count run.forward.rows run.kernel.lseValues)
    (peerLoss runs rows) (peerLSE runs rows)
    ((locals.losses.values row localLossBound).symm.trans peer.1)
    ((locals.lse.values row localLSEBound).symm.trans peer.2)
    (fun r rb => (transmitted_peer_values obs runs world rows cols o x targets group r row rb rowBound).1)
    (fun r rb => (transmitted_peer_values obs runs world rows cols o x targets group r row rb rowBound).2)
  have lossExpression : aggregateLossValue arithmetic run.options run.forward.world run.forward.count run.forward.rows
      run.kernel.lossValues run.kernel.lseValues (peerLoss runs rows) (peerLSE runs rows) (obs.integer run.initial run.labels) row=
      ↑(Spec.loss (world*cols) o.logitScale o.smoothing o.squareScale (x row) (targets row) o.ignored) := by
    simpa only [aggregateLossValue,correctionLoss,correctionZ,run,group.worldEq rank rankBound,group.rowsEq rank rankBound,
      group.labels rank rankBound row rowBound,group.ignoredEq rank rankBound,group.scaleEq rank rankBound] using math.1
  have lseExpression : rankResult arithmetic.logsumexp run.forward.world run.forward.rows
      (localResult arithmetic.logsumexp run.forward.count run.forward.rows run.kernel.lseValues) (peerLSE runs rows) row=
      ↑(Spec.lse (world*cols) o.logitScale (x row)) := by
    simpa only [run,group.worldEq rank rankBound,group.rowsEq rank rankBound] using math.2.1
  have zExpression : correctionZ arithmetic run.options (obs.integer run.initial run.labels row)
      (rankResult arithmetic.logsumexp run.forward.world run.forward.rows
        (localResult arithmetic.logsumexp run.forward.count run.forward.rows run.kernel.lseValues) (peerLSE runs rows) row)=
      ↑(Spec.zLoss (world*cols) o.logitScale o.squareScale (x row) (targets row) o.ignored) := by
    simpa only [correctionZ,run,group.worldEq rank rankBound,group.rowsEq rank rankBound,
      group.labels rank rankBound row rowBound,group.ignoredEq rank rankBound,group.scaleEq rank rankBound] using math.2.2
  have lossBound : row<run.returned.losses.numel := by simpa [Tensor.numel,values.1.shapeEq] using hr
  have lseBound : row<run.returned.context.lse.numel := by simpa [Tensor.numel,values.2.1.shapeEq] using hr
  have zBound : row<run.returned.zLosses.numel := by simpa [Tensor.numel,values.2.2.1.shapeEq] using hr
  exact ⟨(values.1.values row lossBound).trans lossExpression,(values.2.1.values row lseBound).trans lseExpression,
    (values.2.2.1.values row zBound).trans zExpression⟩

#axiomsClean actual_peer_values
#axiomsClean transmitted_peer_values
#axiomsClean returned_split
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Host
