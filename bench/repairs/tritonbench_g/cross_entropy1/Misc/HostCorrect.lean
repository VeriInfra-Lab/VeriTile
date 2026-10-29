import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Contract
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Entries
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.CallerBytes
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.StorageWitness
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.RoleBindings

namespace VeriTile.Bench.Repairs.CrossEntropy1.Correct
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

/-- Public single-rank/unsplit Real entry. Both traces are extracted from
actual successful wrapper calls; backward retrieves the context captured by
this forward call. Saved LSE and the returned gradient are proved internally. -/
theorem unsplit {σ : Type} (obs : Observations (WithBot ℝ) σ)
    (call : Entry.Invocation obs RealSource.algebra)
    (ops : TensorLibrary obs RealHost.arithmetic call.runtime) (ints : IntegerViews obs)
    (comms : CollectiveLibrary obs RealHost.arithmetic call.runtime call.groups.size call.groups.rank)
    (dtype : call.logits.dtype="torch.float32") (labelDtype : call.labels.dtype="torch.int64")
    (single : ¬(call.data.world>1 ∨ call.data.cols>65536)) (smooth : 0≤call.options.smoothing)
    (x : Nat → Nat → ℝ)
    (caller : ∀ row, row<call.data.rows → ∀ col, col<call.data.cols →
      value obs call.initial call.logits (row*call.data.cols+col)=↑(x row col))
    (valid : ∀ row, row<call.data.rows → obs.integer call.initial call.labels row=call.options.ignored ∨
      (0≤obs.integer call.initial call.labels row ∧ obs.integer call.initial call.labels row<(call.data.world*call.data.cols : Nat)))
    (versions : σ → Nat → Nat) (before after : World (WithBot ℝ) σ)
    (between : SavedPreserves obs call.returned.context call.saved before) (callerUpstream output : Tensor)
    (upstreamLive : before.live callerUpstream.region) (upstreamShape : callerUpstream.shape=[call.data.rows])
    (upstreamDtype : callerUpstream.dtype="torch.float32") (upstream : Nat → ℝ)
    (upstreamValues : ∀ row, row<call.data.rows → value obs before callerUpstream row=↑(upstream row))
    (success : Autograd.backwardEntry RealSource.algebra call.runtime versions
      (Autograd.capture versions call.returned.context call.saved.native) callerUpstream before=.ok output after) :
    (∀ row, row<call.data.rows → value obs call.saved call.returned.losses row=
      ↑(Math.loss (call.data.world*call.data.cols) (x row) (obs.integer call.initial call.labels row)
        call.options.ignored call.options.smoothing call.options.squareScale)) ∧
    (∀ row, row<call.data.rows → ∀ col, col<call.data.cols → value obs after output (row*call.data.cols+col)=
      ↑(Math.gradient (call.data.world*call.data.cols) (x row) (obs.integer call.initial call.labels row)
        call.options.ignored call.options.smoothing call.options.squareScale (upstream row) col)) := by
  have group := forward_group_metadata obs call.runtime call.groups call.data
  exact RealHost.complete_unsplit obs call.runtime call.library ops ints call.objects call.groups.size call.groups.rank comms
    call.kernel call.views dtype labelDtype group.1 group.2.1 single smooth x caller valid between
    callerUpstream output upstreamLive upstreamShape upstreamDtype upstream upstreamValues
    (call.backwardData versions callerUpstream output before after success)

def runs {σ : Type} {obs : Observations (WithBot ℝ) σ}
    (calls : Nat → Entry.Invocation obs RealSource.algebra)
    (ops : ∀ r, TensorLibrary obs RealHost.arithmetic (calls r).runtime)
    (splits : ∀ r, (calls r).data.world>1 ∨ (calls r).data.cols>65536) : Nat → RealHost.RankRun σ obs :=
  fun r => (calls r).realRun (ops r) (splits r)

/-- Public local-split/distributed Real entry. Each peer starts with an actual
successful wrapper call. Collective payloads are the actual peer-local
tensors, whose values are derived by the complete theorem, not expected-output
assumptions. This includes local splitting at the same time as rank reduction. -/
theorem split {σ : Type} (obs : Observations (WithBot ℝ) σ)
    (calls : Nat → Entry.Invocation obs RealSource.algebra)
    (ops : ∀ r, TensorLibrary obs RealHost.arithmetic (calls r).runtime)
    (splits : ∀ r, (calls r).data.world>1 ∨ (calls r).data.cols>65536)
    (world rows cols : Nat) (o : Options) (x : Nat → Nat → ℝ) (targets : Nat → Int)
    (group : RealHost.CoherentRanks obs (runs calls ops splits) world rows cols o x targets)
    (rank : Nat) (rankBound : rank<world) (ints : IntegerViews obs)
    (comms : CollectiveLibrary obs RealHost.arithmetic (calls rank).runtime (calls rank).groups.size (calls rank).groups.rank)
    (binding : RankInputs comms ((calls rank).splitData (splits rank)).rankCalls
      (RealHost.peerLoss (runs calls ops splits) rows) (RealHost.peerLSE (runs calls ops splits) rows))
    (versions : σ → Nat → Nat) (before after : World (WithBot ℝ) σ)
    (between : SavedPreserves obs (calls rank).returned.context (calls rank).saved before) (callerUpstream output : Tensor)
    (upstreamLive : before.live callerUpstream.region) (upstreamShape : callerUpstream.shape=[rows])
    (upstreamDtype : callerUpstream.dtype="torch.float32") (upstream : Nat → ℝ)
    (upstreamValues : ∀ row, row<rows → value obs before callerUpstream row=↑(upstream row))
    (success : Autograd.backwardEntry RealSource.algebra (calls rank).runtime versions
      (Autograd.capture versions (calls rank).returned.context (calls rank).saved.native) callerUpstream before=.ok output after) :
    (∀ row, row<rows → value obs (calls rank).saved (calls rank).returned.losses row=
      ↑(Math.loss (world*cols) (x row) (targets row) o.ignored o.smoothing o.squareScale)) ∧
    (∀ row, row<rows → ∀ col, col<cols → value obs after output (row*cols+col)=
      ↑(Math.gradient (world*cols) (x row) (targets row) o.ignored o.smoothing o.squareScale (upstream row) (rank*cols+col))) := by
  exact RealHost.complete_split obs (runs calls ops splits) world rows cols o x targets group rank rankBound ints
    (calls rank).groups.size (calls rank).groups.rank comms (calls rank).size binding
    (calls rank).objects (calls rank).views between callerUpstream output upstreamLive upstreamShape upstreamDtype upstream upstreamValues
    ((calls rank).backwardData versions callerUpstream output before after success)

theorem successful_classes_positive {α σ : Type} [Inhabited α] {obs : Observations α σ} {M : FP32 α}
    (call : Entry.Invocation obs M) : 0<call.data.cols := by
  have nonzero := call.data.nonzero
  by_contra no
  have zero : call.data.cols=0 := by omega
  simp [zero,fwdWidth,nextPower] at nonzero

theorem invalid_labels_shape {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (o : Options) (logits labels : Tensor) (s : World α σ) (rows cols : Nat)
    (shape : logits.shape=[rows,cols]) (wrong : labels.shape≠[rows]) :
    Host.forward M rt o logits labels s=.error "labels shape assertion" s := by
  simp only [Host.forward,shape,if_neg wrong]

theorem checked_real {σ : Type} (obs : Observations (WithBot ℝ) σ)
    (call : Entry.Invocation obs RealSource.algebra)
    (ops : TensorLibrary obs RealHost.arithmetic call.runtime) (ints : IntegerViews obs)
    (comms : CollectiveLibrary obs RealHost.arithmetic call.runtime call.groups.size call.groups.rank)
    (metadata : Checked.ForwardMetadata)
    (admission : Checked.InputContract metadata call.options call.logits call.labels (obs.integer call.initial call.labels))
    (x : Nat → Nat → ℝ)
    (caller : ∀ row, row<call.data.rows → ∀ col, col<call.data.cols →
      value obs call.initial call.logits (row*call.data.cols+col)=↑(x row col))
    (versions : σ → Nat → Nat) (before after : World (WithBot ℝ) σ)
    (between : SavedPreserves obs call.returned.context call.saved before) (callerUpstream output : Tensor)
    (upstreamLive : before.live callerUpstream.region) (upstreamShape : callerUpstream.shape=[call.data.rows])
    (upstreamMetadata : Checked.BackwardMetadata)
    (upstreamAdmission : Checked.BackwardContract upstreamMetadata call.data.rows call.returned.context.logits callerUpstream)
    (upstream : Nat → ℝ)
    (upstreamValues : ∀ row, row<call.data.rows → value obs before callerUpstream row=↑(upstream row))
    (success : Autograd.backwardEntry RealSource.algebra call.runtime versions
      (Autograd.capture versions call.returned.context call.saved.native) callerUpstream before=.ok output after) :
    (∀ row, row<call.data.rows → value obs call.saved call.returned.losses row=
      ↑(Math.loss (call.data.world*call.data.cols) (x row) (obs.integer call.initial call.labels row)
        call.options.ignored call.options.smoothing call.options.squareScale)) ∧
    (∀ row, row<call.data.rows → ∀ col, col<call.data.cols → value obs after output (row*call.data.cols+col)=
      ↑(Math.gradient (call.data.world*call.data.cols) (x row) (obs.integer call.initial call.labels row)
        call.options.ignored call.options.smoothing call.options.squareScale (upstream row) col)) := by
  have inputs := Checked.invocation_inputs call metadata admission
  have world := inputs.2.2.1
  have rank := inputs.2.2.2.1
  have dtype := inputs.2.2.2.2.1
  have labelDtype := inputs.2.2.2.2.2.1
  have smooth := inputs.2.2.2.2.2.2.1
  have valid := inputs.2.2.2.2.2.2.2
  have upstreamDtype : callerUpstream.dtype="torch.float32" := by
    unfold Checked.BackwardContract at upstreamAdmission; tauto
  by_cases small : call.data.cols≤65536
  · have single : ¬(call.data.world>1 ∨ call.data.cols>65536) := by omega
    exact unsplit obs call ops ints comms dtype labelDtype single smooth x caller valid
      versions before after between callerUpstream output upstreamLive upstreamShape upstreamDtype upstream upstreamValues success
  · have splitFlag : call.data.world>1 ∨ call.data.cols>65536 := Or.inr (by omega)
    let calls := fun (_ : Nat) => call
    let operations := fun (_ : Nat) => ops
    let flags := fun (_ : Nat) => splitFlag
    have group : RealHost.CoherentRanks obs (runs calls operations flags) 1 call.data.rows call.data.cols
        call.options x (obs.integer call.initial call.labels) := by
      refine ⟨by decide,successful_classes_positive call,smooth,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
      · intro r hr; exact world
      · intro r hr
        have : r=0 := by omega
        simpa only [this] using rank
      · intro r hr; rfl
      · intro r hr; rfl
      · intro r hr; rfl
      · intro r hr; rfl
      · intro r hr; rfl
      · intro r hr; exact dtype
      · intro r hr; exact labelDtype
      · intro r hr row rowBound col colBound
        have : r=0 := by omega
        simpa only [this,Nat.zero_mul,Nat.zero_add] using caller row rowBound col colBound
      · intro r hr row rowBound; rfl
      · simpa only [world,Nat.one_mul] using valid
    have binding : RankInputs comms (call.splitData splitFlag).rankCalls
        (RealHost.peerLoss (runs calls operations flags) call.data.rows)
        (RealHost.peerLSE (runs calls operations flags) call.data.rows) := by
      exact Checked.rank_inputs_single comms (call.splitData splitFlag).rankCalls _ _ (by omega)
    have result := split obs calls operations flags 1 call.data.rows call.data.cols call.options x
      (obs.integer call.initial call.labels) group 0 (by decide) ints comms binding versions before after between
      callerUpstream output upstreamLive upstreamShape upstreamDtype upstream upstreamValues success
    simpa only [calls,world,Nat.one_mul,Nat.zero_mul,Nat.zero_add] using result

#axiomsClean checked_real

#axiomsClean unsplit
#axiomsClean split
#axiomsClean successful_classes_positive
#axiomsClean invalid_labels_shape
#specNonCircular Math.loss avoiding [Source.forward,Source.backward,Grid.forward,Grid.backward,Host.forward,Host.backward]
#specNonCircular Math.gradient avoiding [Source.forward,Source.backward,Grid.forward,Grid.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Correct
