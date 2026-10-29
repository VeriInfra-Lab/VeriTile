import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.Contract
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.WholeUnsplit
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.WholeSplit

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Correct
open VeriTile.Triton Host
open CrossEntropy1.Host (World Observations TensorLibrary IntegerViews CollectiveLibrary SavedPreserves value RankInputs)
open CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 10000000

/-- Checked complete local call, including local splitting at large widths.
The only peer family used below is constructed from this actual invocation. -/
theorem checked_real {σ : Type} (obs : Observations (WithBot ℝ) σ)
    (call : Entry.Invocation obs RealSource.algebra)
    (ops : TensorLibrary obs CrossEntropy1.RealHost.arithmetic call.runtime.toRuntime)
    (zeros : ZeroAPI obs call.runtime CrossEntropy1.RealHost.arithmetic.zero)
    (ints : IntegerViews obs)
    (comms : CollectiveLibrary obs CrossEntropy1.RealHost.arithmetic call.runtime.toRuntime call.groups.size call.groups.rank)
    (metadata : Checked.ForwardMetadata) (scaleFloat scaleFinite : Bool)
    (admission : Checked.InputContract metadata scaleFloat scaleFinite call.options call.logits call.labels
      (obs.integer call.initial call.labels))
    (versions : σ → Nat → Nat) (upstream output : Tensor) (zUpstream : Option Tensor)
    (before after : World (WithBot ℝ) σ)
    (between : SavedPreserves obs call.returned.context.base call.saved before)
    (upstreamLive : before.live upstream.region) (upstreamMetadata : Checked.BackwardMetadata)
    (upstreamAdmission : Checked.BackwardContract upstreamMetadata call.data.rows call.returned.context.logits upstream)
    (x : Nat → Nat → ℝ) (g : Nat → ℝ)
    (caller : ∀ row, row<call.data.rows → ∀ col, col<call.data.cols →
      value obs call.initial call.logits (row*call.data.cols+col)=↑(x row col))
    (upstreams : ∀ row, row<call.data.rows → value obs before upstream row=↑(g row))
    (success : backwardEntry RealSource.algebra call.runtime versions call.returned.context call.saved.native
      upstream zUpstream before=.ok output after) :
    Checked.forward RealSource.algebra call.runtime metadata scaleFloat scaleFinite call.options call.logits call.labels
      (obs.integer call.initial call.labels) call.initial=.ok call.returned call.saved ∧
    Checked.backward RealSource.algebra call.runtime versions call.returned.context call.saved.native
      upstreamMetadata call.data.rows upstream zUpstream before=.ok output after ∧
    output.shape=[call.data.rows,call.data.cols] ∧
    (call.options.inplace=Bool.true → output=call.returned.context.inplaceLogits) ∧
    ∀ row, row<call.data.rows →
      value obs call.saved call.returned.losses row=
        ↑(Spec.loss call.data.cols call.options.logitScale call.options.smoothing call.options.squareScale
          (x row) (obs.integer call.initial call.labels row) call.options.ignored) ∧
      value obs call.saved call.returned.zLosses row=
        ↑(Spec.zLoss call.data.cols call.options.logitScale call.options.squareScale
          (x row) (obs.integer call.initial call.labels row) call.options.ignored) ∧
      ∀ col, col<call.data.cols → value obs after output (row*call.data.cols+col)=
        ↑(Spec.gradient call.data.cols call.options.logitScale call.options.smoothing call.options.squareScale
          (g row) (x row) (obs.integer call.initial call.labels row) call.options.ignored col) := by
  refine ⟨Checked.source_success call metadata scaleFloat scaleFinite admission,
    Checked.backward_success.mpr ⟨upstreamAdmission,success⟩,?_⟩
  have inputs := Checked.invocation_inputs call metadata scaleFloat scaleFinite admission
  have world := inputs.2.2.1
  have rank := inputs.2.2.2.1
  have dtype := inputs.2.2.2.2.1
  have labelDtype := inputs.2.2.2.2.2.1
  have smooth := inputs.2.2.2.2.2.2.1
  have process := inputs.2.2.2.2.2.2.2.1
  have targets := inputs.2.2.2.2.2.2.2.2
  have upstreamShape : upstream.shape=[call.data.rows] := by
    unfold Checked.BackwardContract CrossEntropy1.Checked.BackwardContract at upstreamAdmission; tauto
  have upstreamDtype : upstream.dtype="torch.float32" := by
    unfold Checked.BackwardContract CrossEntropy1.Checked.BackwardContract at upstreamAdmission; tauto
  by_cases small : call.data.cols≤65536
  · have single : ∀ group, call.options.processGroup=some group → call.groups.size group=1 := by
      intro group h; simp [process] at h
    have result := unsplit_forward_backward obs call.runtime call.library call.groups ints call.objects call.geometry
      versions call.options call.logits call.labels upstream output zUpstream call.initial call.saved before after call.returned
      between call.data.rows call.data.cols call.data.logitsShape single small dtype labelDtype call.logitsLive call.labelsLive
      call.rank call.registered upstreamLive upstreamShape upstreamDtype smooth x g caller targets upstreams call.success success
    exact result.2.2
  · have splitFlag : call.data.world>1 ∨ call.data.cols>65536 := Or.inr (by omega)
    let run : RankRun σ obs := ⟨call.runtime,call.library,ops,zeros,call.options,call.logits,call.labels,
      call.initial,call.saved,call.returned,call.data,call.kernel,call.splitData splitFlag,splitFlag⟩
    let runs := fun (_ : Nat) => run
    have positive : 0<call.data.cols := by omega
    have group : CoherentRanks obs runs 1 call.data.rows call.data.cols call.options x
        (obs.integer call.initial call.labels) := by
      refine ⟨by decide,positive,smooth,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
      · intro r hr; exact world
      · intro r hr
        have : r=0 := by omega
        simpa only [this] using rank
      · intro r hr; rfl
      · intro r hr; rfl
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
      · simpa only [Nat.one_mul] using targets
    have binding : RankInputs comms (call.splitData splitFlag).rankCalls
        (peerLoss runs call.data.rows) (peerLSE runs call.data.rows) :=
      CrossEntropy1.Checked.rank_inputs_single comms _ _ _ (by omega)
    have result := split_forward_backward obs runs 1 call.data.rows call.data.cols call.options x
      (obs.integer call.initial call.labels) group 0 (by decide) ints call.groups.size call.groups.rank comms call.size binding
      call.objects call.geometry call.logitsLive call.rank call.registered versions upstream output zUpstream before after
      between upstreamLive upstreamShape upstreamDtype g upstreams success
    simpa only [runs,run,Nat.one_mul,Nat.zero_mul,Nat.zero_add] using result

#axiomsClean checked_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Correct
