import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.WholeGraph
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.Entries

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Host
open VeriTile.Triton
open CrossEntropy1.Source (FP32 column)
open CrossEntropy1.Host (World Step Observations ObjectRegistry Preserves TensorArithmetic TensorLibrary IntegerViews CollectiveLibrary lastStride)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000

theorem backward_written_coordinate {α σ : Type} [Inhabited α] {obs : Observations α σ} {rt : Runtime α σ}
    {objects : ObjectRegistry rt.toRuntime} {M : FP32 α} {c : Context} {callerUpstream output : Tensor}
    {initial final : World α σ} {e : BackwardData M rt c callerUpstream output initial final}
    (k : BackwardKernelData obs objects e) (q : CrossEntropy1.Grid.Task) (present : q∈CrossEntropy1.Grid.tasks e.rows e.count)
    (r : RegionName) (i : Nat) (written : Grid.backwardWrites e.parameters q (r,i)) :
    ∃ j, j<e.work.numel ∧ r=e.work.region ∧ i=e.work.index j := by
  obtain ⟨lane,active,address⟩ := written
  have hr := ((CrossEntropy1.Grid.task_mem _ _ _).mp present).1
  let col := column e.parameters.toParameters q.2 lane.val
  have hc : col<e.cols := active
  have shape : e.work.shape=[e.rows,e.cols] := k.work.buffer.shape.trans e.shape
  have bound : q.1*e.cols+col<e.work.numel := by
    simp only [Tensor.numel,shape,List.prod_cons,List.prod_nil,Nat.mul_one]
    have product := Nat.mul_le_mul_right e.cols (Nat.succ_le_of_lt hr)
    simp only [Nat.succ_mul] at product
    omega
  refine ⟨q.1*e.cols+col,bound,congrArg Prod.fst address,?_⟩
  exact (congrArg Prod.snd address).trans
    (k.work.layout e.rows e.cols shape q.1 col hr hc).symm

/-- Complete logical frame of the actual backward return. It includes the
in-place work view and, if distinct, the exact copyback destination. -/
theorem backward_return_frame {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : LocalAPI obs rt) {objects : ObjectRegistry rt.toRuntime} {M : FP32 α} {c : Context}
    {callerUpstream output : Tensor} {initial final : World α σ}
    {e : BackwardData M rt c callerUpstream output initial final} (k : BackwardKernelData obs objects e)
    (r : RegionName) (i : Nat) (live : initial.live r)
    (outsideWork : r≠e.work.region ∨ ∀ j, j<e.work.numel → i≠e.work.index j)
    (outsideOutput : r≠output.region ∨ ∀ j, j<output.numel → i≠output.index j) :
    final.memory r i=initial.memory r i := by
  have frame := Grid.backward_complete_frame M e.parameters k.launch.ready.memory e.count k.separation r i (by
    intro q present written
    obtain ⟨j,hj,region,index⟩ := backward_written_coordinate k q present r i written
    rcases outsideWork with outside|outside
    · exact outside region
    · exact outside j hj index)
  have exit := lib.tensor.exit _ _ _ k.launch.exitCall
  have computed : e.computed.memory r i=initial.memory r i :=
    (exit.memory r i (k.initialFrame.live r live)).trans (frame.trans (k.initialFrame.memory r i live))
  rcases e.finish with ⟨same,state⟩|⟨different,returned,call⟩
  · simpa only [state] using computed
  · have copied := lib.tensor.copy output e.work e.computed returned final
      (exit.live _ k.outputLive) (exit.live _ k.workLive) (k.outputShape.trans k.work.buffer.shape.symm) call
    exact (copied.2.2.2 r i outsideOutput).trans computed

/-- Live caller allocations survive the ordinary backward operations. This
does not assume any byte or task-output frame. -/
theorem backward_return_live {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : LocalAPI obs rt) {objects : ObjectRegistry rt.toRuntime} {M : FP32 α} {c : Context}
    {callerUpstream output : Tensor} {initial final : World α σ}
    {e : BackwardData M rt c callerUpstream output initial final} (k : BackwardKernelData obs objects e)
    (r : RegionName) (live : initial.live r) : final.live r := by
  have exit := lib.tensor.exit _ _ _ k.launch.exitCall
  have computed : e.computed.live r := exit.live r (k.initialFrame.live r live)
  rcases e.finish with ⟨same,state⟩|⟨different,returned,call⟩
  · simpa only [state] using computed
  · exact (lib.tensor.copy output e.work e.computed returned final
      (exit.live _ k.outputLive) (exit.live _ k.workLive)
      (k.outputShape.trans k.work.buffer.shape.symm) call).2.1 r computed

theorem backward_out_of_place_fresh {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : LocalAPI obs rt) {objects : ObjectRegistry rt.toRuntime} {M : FP32 α} {c : Context}
    {callerUpstream output : Tensor} {initial final : World α σ}
    {e : BackwardData M rt c callerUpstream output initial final} (k : BackwardKernelData obs objects e)
    (logitsLive : initial.live c.logits.region) (outOfPlace : c.options.inplace≠Bool.true) :
    ¬initial.live output.region ∧ ¬initial.live e.work.region := by
  have firstCall : rt.emptyLike c.logits Bool.false e.upstreamReady=.ok e.first e.allocated := by
    simpa only [if_neg outOfPlace] using e.firstCall
  have first := lib.tensor.emptyLike _ _ _ _ _ (k.snapshot.materialized.frame.live _ logitsLive) firstCall
  have freshFirst : ¬initial.live e.first.region := fun live => first.1.fresh (k.snapshot.materialized.frame.live _ live)
  have outputEq : output=e.first := by simpa only [if_neg outOfPlace] using e.outputIdentity
  refine ⟨by simpa only [outputEq] using freshFirst,?_⟩
  by_cases separate : lastStride e.first≠1
  · have call : rt.emptyLike c.logits Bool.true e.allocated=.ok e.work e.prepared := by
      simpa only [if_pos separate] using e.workCall
    have allocatedFrame := k.snapshot.materialized.frame.trans k.first.frame
    have work := lib.tensor.emptyLike _ _ _ _ _ (allocatedFrame.live _ logitsLive) call
    exact fun live => work.1.fresh (allocatedFrame.live _ live)
  · have workEq : e.work=e.first := by
      have h := e.workCall
      simp only [if_neg separate,Step.ok.injEq] at h
      exact h.1.symm
    simpa only [workEq] using freshFirst

theorem backward_out_of_place_frame {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : LocalAPI obs rt) {objects : ObjectRegistry rt.toRuntime} {M : FP32 α} {c : Context}
    {callerUpstream output : Tensor} {initial final : World α σ}
    {e : BackwardData M rt c callerUpstream output initial final} (k : BackwardKernelData obs objects e)
    (logitsLive : initial.live c.logits.region) (outOfPlace : c.options.inplace≠Bool.true)
    (r : RegionName) (i : Nat) (live : initial.live r) : final.memory r i=initial.memory r i := by
  have fresh := backward_out_of_place_fresh obs rt lib k logitsLive outOfPlace
  exact backward_return_frame obs rt lib k r i live
    (Or.inl (fun same => fresh.2 (same ▸ live))) (Or.inl (fun same => fresh.1 (same ▸ live)))

/-- Source-side whole-forward frame; the split proof already includes
all ordinary aggregation and save/mark calls. This is a value frame, not a
claim that no idempotent native store occurred. -/
theorem forward_return_frame {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (A : TensorArithmetic α) (rt : Runtime α σ) (lib : LocalAPI obs rt)
    (ops : TensorLibrary obs A rt.toRuntime) (zeros : ZeroAPI obs rt A.zero)
    (ints : IntegerViews obs) (gs gr : Nat → Nat)
    (comms : CollectiveLibrary obs A rt.toRuntime gs gr)
    {M : FP32 α} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (labelDtype : labels.dtype="torch.int64") (size : e.world>1 → e.world=gs (o.processGroup.getD 0))
    (aggregation : ForwardAggregation e) (lossPeers lsePeers : Nat → α)
    (inputs : ForwardInputs comms aggregation lossPeers lsePeers) :
    Preserves obs initial final := by
  cases aggregation with
  | direct flag =>
    have ids := unsplit_identity e flag
    have post : Preserves obs e.computed final := by
      rw [←ids.2.1]
      exact (lib.tensor.save _ _ _ _ _ _ e.saveCall).trans (lib.mark _ _ _ e.markCall)
    exact (forward_caller_frame obs rt lib ints k).trans post
  | split flag trace =>
    have frame := returned_split_values obs A rt lib ops zeros ints gs gr comms k
      labelDtype size trace lossPeers lsePeers inputs
    exact k.working.frame.trans frame.2.2.2.2

#axiomsClean forward_return_frame
#axiomsClean backward_written_coordinate
#axiomsClean backward_return_frame
#axiomsClean backward_return_live
#axiomsClean backward_out_of_place_fresh
#axiomsClean backward_out_of_place_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Host
