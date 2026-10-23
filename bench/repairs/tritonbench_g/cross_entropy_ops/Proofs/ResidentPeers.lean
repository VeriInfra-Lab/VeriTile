import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.ResidentValues

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Resident
open VeriTile.Triton Host
open CrossEntropy1.Floating
open CrossEntropy1.Physical
open CrossEntropy1.Host (World Observations TensorLibrary CollectiveLibrary localResult local_reduction_values value)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

/-- Actual successful peer invocations; only metadata and ordinary operation
contracts are fields. Peer payload values are derived below. -/
structure PeerCalls {σ : Type} (obs : Observations SignedExpr σ) (world rows cols : Nat) where
  calls : Nat → Entry.Invocation obs signedFP
  operations : ∀ rank, TensorLibrary obs signedHost (calls rank).runtime.toRuntime
  splits : ∀ rank, (calls rank).data.world>1 ∨ (calls rank).data.cols>65536
  rowsEq : ∀ rank, rank<world → (calls rank).data.rows=rows
  colsEq : ∀ rank, rank<world → (calls rank).data.cols=cols
  worldEq : ∀ rank, rank<world → (calls rank).data.world=world
  rankEq : ∀ rank, rank<world → (calls rank).data.rank=rank

def PeerCalls.aggregate {σ : Type} {obs : Observations SignedExpr σ} {world rows cols}
    (group : PeerCalls obs world rows cols) (rank : Nat) :=
  (group.calls rank).splitData (group.splits rank)

def PeerCalls.loss {σ : Type} {obs : Observations SignedExpr σ} {world rows cols}
    (group : PeerCalls obs world rows cols) (i : Nat) : SignedExpr :=
  let a := group.aggregate (i/rows)
  value obs a.localState a.localPair.1 (i%rows)

def PeerCalls.lse {σ : Type} {obs : Observations SignedExpr σ} {world rows cols}
    (group : PeerCalls obs world rows cols) (i : Nat) : SignedExpr :=
  let a := group.aggregate (i/rows)
  value obs a.localState a.localPair.2 (i%rows)

theorem transmitted_sources {σ : Type} {obs : Observations SignedExpr σ} {world rows cols}
    (group : PeerCalls obs world rows cols) (rank row : Nat) (rankBound : rank<world) (rowBound : row<rows) :
    group.loss (rank*rows+row)=localResult signedHost.sum (group.calls rank).data.count rows (group.calls rank).kernel.lossValues row ∧
    group.lse (rank*rows+row)=localResult signedHost.logsumexp (group.calls rank).data.count rows (group.calls rank).kernel.lseValues row := by
  let call := group.calls rank
  let a := call.splitData (group.splits rank)
  have arrays := forward_arrays obs call.runtime call.library call.kernel
  have locals := local_reduction_values obs signedHost call.runtime.toRuntime (group.operations rank) call.data.count call.data.rows
    call.data.losses call.data.lse call.data.computed a.localPair a.localState call.kernel.lossValues call.kernel.lseValues
    arrays.1 arrays.2 call.kernel.separation.outputDistinct.symm a.localCalls
  have inside : row<call.data.rows := by simpa only [call,group.rowsEq rank rankBound] using rowBound
  have first := locals.losses.values row (by simpa [Tensor.numel,locals.losses.shapeEq] using inside)
  have second := locals.lse.values row (by simpa [Tensor.numel,locals.lse.shapeEq] using inside)
  have coordinates := CrossEntropy1.RealHost.block_coordinates rows row rank rowBound
  unfold PeerCalls.loss PeerCalls.lse
  rw [coordinates.1,coordinates.2]
  simpa only [call,a,group.rowsEq rank rankBound] using And.intro first second

inductive PeerMessages {σ : Type} {obs : Observations SignedExpr σ} (call : Entry.Invocation obs signedFP) where
  | direct (single : ¬(call.data.world>1 ∨ call.data.cols>65536))
  | split (split : call.data.world>1 ∨ call.data.cols>65536)
      (group : PeerCalls obs call.data.world call.data.rows call.data.cols)

def PeerMessages.loss {σ : Type} {obs : Observations SignedExpr σ} {call : Entry.Invocation obs signedFP} : PeerMessages call → Nat → SignedExpr
  | .direct _ => fun _ => ⟨.raw 0,Bool.false⟩
  | .split _ group => group.loss

def PeerMessages.lse {σ : Type} {obs : Observations SignedExpr σ} {call : Entry.Invocation obs signedFP} : PeerMessages call → Nat → SignedExpr
  | .direct _ => fun _ => ⟨.raw 0,Bool.false⟩
  | .split _ group => group.lse

def PeerMessages.aggregation {σ : Type} {obs : Observations SignedExpr σ} {call : Entry.Invocation obs signedFP} :
    PeerMessages call → ForwardAggregation call.data
  | .direct single => .direct single
  | .split flag _ => .split flag (call.splitData flag)

/-- Every rank belongs to this same collective invocation. The local peer
aliases the main invocation and all source scalar metadata agrees. -/
def PeerMessages.Compatible {σ : Type} {obs : Observations SignedExpr σ} {call : Entry.Invocation obs signedFP} : PeerMessages call → Prop
  | .direct _ => True
  | .split _ group => group.calls call.data.rank=call ∧ ∀ rank, rank<call.data.world →
      (group.calls rank).options.processGroup=call.options.processGroup ∧
      (group.calls rank).options.smoothing=call.options.smoothing ∧
      (group.calls rank).options.squareScale=call.options.squareScale ∧
      (group.calls rank).options.logitScale=call.options.logitScale ∧
      (group.calls rank).options.ignored=call.options.ignored

/-- Peer values come from the actual local source traces. This is a
conclusion, not an expected-output or saved-LSE premise. -/
def PeerMessages.SourceValues {σ : Type} {obs : Observations SignedExpr σ}
    {call : Entry.Invocation obs signedFP} : PeerMessages call → Prop
  | .direct _ => True
  | .split _ group => ∀ rank row, rank<call.data.world → row<call.data.rows →
      group.loss (rank*call.data.rows+row)=localResult signedHost.sum (group.calls rank).data.count call.data.rows
        (group.calls rank).kernel.lossValues row ∧
      group.lse (rank*call.data.rows+row)=localResult signedHost.logsumexp (group.calls rank).data.count call.data.rows
        (group.calls rank).kernel.lseValues row

theorem PeerMessages.sourceValues {σ : Type} {obs : Observations SignedExpr σ}
    {call : Entry.Invocation obs signedFP} (messages : PeerMessages call) : messages.SourceValues := by
  cases messages with
  | direct single => trivial
  | split flag group => exact transmitted_sources group

#axiomsClean transmitted_sources
#axiomsClean PeerMessages.sourceValues
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Resident
