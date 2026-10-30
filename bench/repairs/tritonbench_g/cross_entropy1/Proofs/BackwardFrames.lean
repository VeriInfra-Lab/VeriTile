import bench.repairs.tritonbench_g.cross_entropy1.Proofs.BackwardKernel

namespace VeriTile.Bench.Repairs.CrossEntropy1.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000

theorem backward_written_coordinate {α σ : Type} [Inhabited α] {obs : Observations α σ} {rt : Runtime α σ}
    {objects : ObjectRegistry rt} {M : FP32 α} {c : Context} {callerUpstream output : Tensor}
    {initial final : World α σ} {e : BackwardData M rt c callerUpstream output initial final}
    (k : BackwardKernelData obs objects e) (q : Grid.Task) (present : q∈Grid.tasks e.rows e.count)
    (r : RegionName) (i : Nat) (written : Grid.backwardWrites e.parameters q (r,i)) :
    ∃ j, j<e.work.numel ∧ r=e.work.region ∧ i=e.work.index j := by
  obtain ⟨lane,active,address⟩ := written
  have hr := ((Grid.task_mem _ _ _).mp present).1
  let col := column e.parameters q.2 lane.val
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
    (lib : Library obs rt) {objects : ObjectRegistry rt} {M : FP32 α} {c : Context}
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
  have exit := lib.exit _ _ _ k.launch.exitCall
  have computed : e.computed.memory r i=initial.memory r i :=
    (exit.memory r i (k.initialFrame.live r live)).trans (frame.trans (k.initialFrame.memory r i live))
  rcases e.finish with ⟨same,state⟩|⟨different,returned,call⟩
  · simpa only [state] using computed
  · have copied := lib.copy output e.work e.computed returned final
      (exit.live _ k.outputLive) (exit.live _ k.workLive) (k.outputShape.trans k.work.buffer.shape.symm) call
    exact (copied.2.2.2 r i outsideOutput).trans computed

theorem backward_out_of_place_fresh {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : Library obs rt) {objects : ObjectRegistry rt} {M : FP32 α} {c : Context}
    {callerUpstream output : Tensor} {initial final : World α σ}
    {e : BackwardData M rt c callerUpstream output initial final} (k : BackwardKernelData obs objects e)
    (logitsLive : initial.live c.logits.region) (outOfPlace : c.options.inplace≠Bool.true) :
    ¬initial.live output.region ∧ ¬initial.live e.work.region := by
  have firstCall : rt.emptyLike c.logits Bool.false e.upstreamReady=.ok e.first e.allocated := by
    simpa only [if_neg outOfPlace] using e.firstCall
  have first := lib.emptyLike _ _ _ _ _ (k.snapshot.materialized.frame.live _ logitsLive) firstCall
  have freshFirst : ¬initial.live e.first.region := fun live => first.1.fresh (k.snapshot.materialized.frame.live _ live)
  have outputEq : output=e.first := by simpa only [if_neg outOfPlace] using e.outputIdentity
  refine ⟨by simpa only [outputEq] using freshFirst,?_⟩
  by_cases separate : lastStride e.first≠1
  · have call : rt.emptyLike c.logits Bool.true e.allocated=.ok e.work e.prepared := by
      simpa only [if_pos separate] using e.workCall
    have allocatedFrame := k.snapshot.materialized.frame.trans k.first.frame
    have work := lib.emptyLike _ _ _ _ _ (allocatedFrame.live _ logitsLive) call
    exact fun live => work.1.fresh (allocatedFrame.live _ live)
  · have workEq : e.work=e.first := by
      have h := e.workCall
      simp only [if_neg separate,Step.ok.injEq] at h
      exact h.1.symm
    simpa only [workEq] using freshFirst

theorem backward_out_of_place_frame {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : Library obs rt) {objects : ObjectRegistry rt} {M : FP32 α} {c : Context}
    {callerUpstream output : Tensor} {initial final : World α σ}
    {e : BackwardData M rt c callerUpstream output initial final} (k : BackwardKernelData obs objects e)
    (logitsLive : initial.live c.logits.region) (outOfPlace : c.options.inplace≠Bool.true)
    (r : RegionName) (i : Nat) (live : initial.live r) : final.memory r i=initial.memory r i := by
  have fresh := backward_out_of_place_fresh obs rt lib k logitsLive outOfPlace
  exact backward_return_frame obs rt lib k r i live
    (Or.inl (fun same => fresh.2 (same ▸ live))) (Or.inl (fun same => fresh.1 (same ▸ live)))

#axiomsClean backward_written_coordinate
#axiomsClean backward_return_frame
#axiomsClean backward_out_of_place_fresh
#axiomsClean backward_out_of_place_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Host
