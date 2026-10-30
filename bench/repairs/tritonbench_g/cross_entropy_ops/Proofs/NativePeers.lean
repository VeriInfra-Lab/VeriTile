import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.WholeGraph
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.NativeSlots
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.NativeLedger

namespace VeriTile.Bench.Repairs.CrossEntropyOps.NativeGraph
open VeriTile.Triton Host
open CrossEntropy1.Floating
open CrossEntropy1.Physical
open CrossEntropy1.Host (World Observations TensorLibrary CollectiveLibrary localResult local_reduction_values value)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

/-- Actual successful peer invocations; only metadata and ordinary operation
contracts are fields. Peer payload values are derived below. -/
structure PeerCalls {σ : Type} (obs : Observations Expr σ) (world rows cols : Nat) where
  calls : Nat → Entry.Invocation obs algebra
  operations : ∀ rank, TensorLibrary obs hostArithmetic (calls rank).runtime.toRuntime
  splits : ∀ rank, (calls rank).data.world>1 ∨ (calls rank).data.cols>65536
  rowsEq : ∀ rank, rank<world → (calls rank).data.rows=rows
  colsEq : ∀ rank, rank<world → (calls rank).data.cols=cols
  worldEq : ∀ rank, rank<world → (calls rank).data.world=world
  rankEq : ∀ rank, rank<world → (calls rank).data.rank=rank

def PeerCalls.aggregate {σ : Type} {obs : Observations Expr σ} {world rows cols}
    (group : PeerCalls obs world rows cols) (rank : Nat) :=
  (group.calls rank).splitData (group.splits rank)

def PeerCalls.loss {σ : Type} {obs : Observations Expr σ} {world rows cols}
    (group : PeerCalls obs world rows cols) (i : Nat) : Expr :=
  let a := group.aggregate (i/rows)
  value obs a.localState a.localPair.1 (i%rows)

def PeerCalls.lse {σ : Type} {obs : Observations Expr σ} {world rows cols}
    (group : PeerCalls obs world rows cols) (i : Nat) : Expr :=
  let a := group.aggregate (i/rows)
  value obs a.localState a.localPair.2 (i%rows)

theorem transmitted_sources {σ : Type} {obs : Observations Expr σ} {world rows cols}
    (group : PeerCalls obs world rows cols) (rank row : Nat) (rankBound : rank<world) (rowBound : row<rows) :
    group.loss (rank*rows+row)=localResult hostArithmetic.sum (group.calls rank).data.count rows (group.calls rank).kernel.lossValues row ∧
    group.lse (rank*rows+row)=localResult hostArithmetic.logsumexp (group.calls rank).data.count rows (group.calls rank).kernel.lseValues row := by
  let call := group.calls rank
  let a := call.splitData (group.splits rank)
  have arrays := forward_arrays obs call.runtime call.library call.kernel
  have locals := local_reduction_values obs hostArithmetic call.runtime.toRuntime (group.operations rank) call.data.count call.data.rows
    call.data.losses call.data.lse call.data.computed a.localPair a.localState call.kernel.lossValues call.kernel.lseValues
    arrays.1 arrays.2 call.kernel.separation.outputDistinct.symm a.localCalls
  have inside : row<call.data.rows := by simpa only [call,group.rowsEq rank rankBound] using rowBound
  have first := locals.losses.values row (by simpa [Tensor.numel,locals.losses.shapeEq] using inside)
  have second := locals.lse.values row (by simpa [Tensor.numel,locals.lse.shapeEq] using inside)
  have coordinates := CrossEntropy1.RealHost.block_coordinates rows row rank rowBound
  unfold PeerCalls.loss PeerCalls.lse
  rw [coordinates.1,coordinates.2]
  simpa only [call,a,group.rowsEq rank rankBound] using And.intro first second

inductive Messages {σ : Type} {obs : Observations Expr σ} (call : Entry.Invocation obs algebra) where
  | direct (single : ¬(call.data.world>1 ∨ call.data.cols>65536))
  | split (split : call.data.world>1 ∨ call.data.cols>65536)
      (group : PeerCalls obs call.data.world call.data.rows call.data.cols)

def Messages.loss {σ : Type} {obs : Observations Expr σ} {call : Entry.Invocation obs algebra} : Messages call → Nat → Expr
  | .direct _ => fun _ => .raw 0
  | .split _ group => group.loss

def Messages.lse {σ : Type} {obs : Observations Expr σ} {call : Entry.Invocation obs algebra} : Messages call → Nat → Expr
  | .direct _ => fun _ => .raw 0
  | .split _ group => group.lse

def Messages.aggregation {σ : Type} {obs : Observations Expr σ} {call : Entry.Invocation obs algebra} :
    Messages call → ForwardAggregation call.data
  | .direct single => .direct single
  | .split flag _ => .split flag (call.splitData flag)

/-- Every rank belongs to this same collective invocation. The local peer
aliases the main invocation and all source scalar metadata agrees. -/
def Messages.Compatible {σ : Type} {obs : Observations Expr σ} {call : Entry.Invocation obs algebra} : Messages call → Prop
  | .direct _ => True
  | .split _ group => group.calls call.data.rank=call ∧ ∀ rank, rank<call.data.world →
      (group.calls rank).options.processGroup=call.options.processGroup ∧
      (group.calls rank).options.smoothing=call.options.smoothing ∧
      (group.calls rank).options.squareScale=call.options.squareScale ∧
      (group.calls rank).options.logitScale=call.options.logitScale ∧
      (group.calls rank).options.ignored=call.options.ignored

def Messages.expectedRank {σ : Type} {obs : Observations Expr σ} {call : Entry.Invocation obs algebra}
    (_messages : Messages call) (location : SourceLocation) : Nat :=
  match location.participant with
  | .main | .backward => call.data.rank
  | .peer rank => rank

def Messages.source {σ : Type} {obs : Observations Expr σ} {call : Entry.Invocation obs algebra}
    (messages : Messages call) (abi : InvocationABI σ) {upstream output before after}
    (b : BackwardData algebra call.runtime call.returned.context upstream output before after)
    (location : SourceLocation) (record : SourceCall σ) : Prop :=
  messages.Compatible ∧ abi.rank location=messages.expectedRank location ∧
  (∀ rank, abi.peerParticipant rank=if rank=call.data.rank then .main else .peer rank) ∧
  match location.participant with
  | .main => record.runtime=call.runtime ∧ ForwardMember call location.slot record
  | .backward => record.runtime=call.runtime ∧ BackwardMember b location.slot record
  | .peer rank => match messages with
    | .direct _ => False
    | .split _ group => rank<call.data.world ∧ rank≠call.data.rank ∧
        record.runtime=(group.calls rank).runtime ∧ ForwardMember (group.calls rank) location.slot record

structure LedgerBoundaries {σ : Type} {obs : Observations Expr σ} {call : Entry.Invocation obs algebra}
    {messages : Messages call} {abi upstream output before after}
    {b : BackwardData algebra call.runtime call.returned.context upstream output before after}
    (ledger : HostLedger (messages.source abi b)) : Prop where
  forwardInitial : ledger.state .main 0=call.initial.native
  forwardFinal : ledger.state .main (ledger.entries .main).length=call.saved.native
  backwardInitial : ledger.state .backward 0=before.native
  backwardFinal : ledger.state .backward (ledger.entries .backward).length=after.native

structure InitialInputViews {σ : Type} (layout : Layout σ) (observe : σ → Nat → Nat → Byte)
    (obs : Observations Expr σ) (call : Entry.Invocation obs algebra) : Prop where
  logits : ∀ row, row<call.data.rows → ∀ col, col<call.data.cols →
    value obs call.initial call.logits (row*call.data.cols+col)=
      .raw (logicalWord layout observe call.initial.native call.logits (row*call.data.cols+col))
  labels : ∀ row, row<call.data.rows → obs.integer call.initial call.labels row=
    logicalInt64 layout observe call.initial.native call.labels row

def Messages.inputViews {σ : Type} {obs : Observations Expr σ} {call : Entry.Invocation obs algebra}
    (messages : Messages call) (layout : Layout σ) (observe : σ → Nat → Nat → Byte) : Prop :=
  match messages with
  | .direct _ => True
  | .split _ group => ∀ rank, rank<call.data.world → InitialInputViews layout observe obs (group.calls rank)

#axiomsClean transmitted_sources
#axiomsClean Messages.source
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.NativeGraph
