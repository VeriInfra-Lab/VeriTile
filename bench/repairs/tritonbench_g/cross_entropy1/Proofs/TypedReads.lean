import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Storage
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.SavedBoundary
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.BackwardReads

namespace VeriTile.Bench.Repairs.CrossEntropy1.Physical
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000

/-- Ordinary contiguous int64 clone semantics, expressed in its actual two
raw FP32-sized byte chunks. This has no cross-entropy or whole-call output. -/
structure CloneBytes {α σ : Type} [Inhabited α] (encode : α → Word)
    (obs : Observations α σ) (rt : Runtime α σ) : Prop where
  labels : ∀ input before output after, before.live input.region → input.isNeg=Bool.false →
    input.dtype="torch.int64" → rt.cloneContiguous input before=.ok output after →
    ∀ i, i < input.numel → rawInt64 encode after output.region i=obs.integer before input i

theorem int_read_region {α σ : Type} [Inhabited α] (encode : α → Word)
    (a b : World α σ) (r : RegionName) (i : Nat)
    (same : ∀ j, b.memory r j=a.memory r j) (coupled : IntRead encode a r i) : IntRead encode b r i := by
  simpa only [IntRead,rawInt64,rawWord,same] using coupled

theorem snapshot_int_reads {α σ : Type} [Inhabited α] (encode : α → Word)
    (obs : Observations α σ) (rt : Runtime α σ) (lib : Library obs rt) (physical : CloneBytes encode obs rt)
    (input output : Tensor) (before after : World α σ) (live : before.live input.region)
    (dtype : input.dtype="torch.int64") (call : snapshot rt input before=.ok output after) :
    ∀ i, i < input.numel → IntRead encode after output.region i := by
  obtain ⟨trace⟩ := snapshot_success rt input output before after call
  have resolved := lib.resolve _ _ _ _ live trace.resolveCall
  have cloned := lib.clone _ _ _ _ resolved.1.live resolved.2 trace.cloneCall
  have intDtype : trace.resolved.dtype="torch.int64" := resolved.1.metadata.dtype.trans dtype
  have raw := physical.labels _ _ _ _ resolved.1.live resolved.2 intDtype trace.cloneCall
  intro i hi
  have bound : i<trace.resolved.numel := by simpa only [resolved.1.metadata.numel] using hi
  exact (cloned.2.2.2 intDtype i bound).trans (raw i bound).symm

/-- Actual forward label loads observe precisely the bytes created by the
preceding snapshot, including when caller labels alias FP32 input storage. -/
theorem forward_label_coupling {α σ : Type} [Inhabited α] (encode : α → Word)
    (obs : Observations α σ) (rt : Runtime α σ) (lib : Library obs rt) (physical : CloneBytes encode obs rt)
    (groups : Groups obs rt) {M : FP32 α} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    {f : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs f)
    (labelsLive : initial.live labels.region) (dtype : labels.dtype="torch.int64")
    (row : Nat) (hr : row<f.rows) :
    Source.label f.parameters k.launch.ready.memory row=rawInt64 encode k.launch.ready f.labels.region row := by
  have group := forward_group_metadata obs rt groups f
  have inside : row<labels.numel := by simpa only [Tensor.numel,f.labelsShape,List.prod_cons,List.prod_nil,Nat.mul_one] using hr
  have initialRead := snapshot_int_reads encode obs rt lib physical labels f.labels f.ranked f.labelsReady
    (group.2.2.live _ labelsLive) dtype f.snapshotCall row inside
  have coupled := int_read_region encode f.labelsReady k.launch.ready f.labels.region row
    (fun j => k.labelsFrame.memory _ j k.labelsSnapshot.materialized.live) initialRead
  have outputPositive : 0<f.labels.numel := by rw [k.labelsSnapshot.materialized.metadata.numel]; omega
  have base := dense_base f.labels k.labelsSnapshot.dense outputPositive
  simpa only [Source.label,ForwardData.parameters,forwardParameters,Region.cast_cast,Region.cast_id,base,Nat.zero_add]
    using coupled

/-- Snapshot byte coherence survives all forward work and the ordinary
between-call preservation boundary. It is derived from clone semantics, not
assumed for the saved context. -/
theorem saved_label_coupling {α σ : Type} [Inhabited α] (encode : α → Word)
    (obs : Observations α σ) (A : TensorArithmetic α) (rt : Runtime α σ)
    (lib : Library obs rt) (physical : CloneBytes encode obs rt) (ops : TensorLibrary obs A rt)
    (ints : IntegerViews obs) (groups : Groups obs rt) (gs gr : Nat → Nat)
    (comms : CollectiveLibrary obs A rt gs gr) {M : FP32 α} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    {f : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs f)
    (logitsLive : initial.live logits.region) (labelsLive : initial.live labels.region)
    (rank : logits.strides.length=logits.shape.length) (dtype : labels.dtype="torch.int64")
    (size : f.world>1 → f.world=gs (o.processGroup.getD 0)) (aggregation : ForwardAggregation f)
    (lossPeers lsePeers : Nat → α) (inputs : ForwardInputs comms aggregation lossPeers lsePeers)
    {before : World α σ} (between : SavedPreserves obs returned.context final before) (row : Nat) (hr : row<f.rows) :
    IntRead encode before returned.context.labels.region row := by
  have prep := forward_preparation obs rt lib groups f logitsLive labelsLive rank
  have group := forward_group_metadata obs rt groups f
  have inside : row<labels.numel := by simpa only [Tensor.numel,f.labelsShape,List.prod_cons,List.prod_nil,Nat.mul_one] using hr
  have read := snapshot_int_reads encode obs rt lib physical labels f.labels f.ranked f.labelsReady
    (group.2.2.live _ labelsLive) dtype f.snapshotCall row inside
  have saved := forward_saved_frame obs A rt lib ops ints gs gr comms k dtype size aggregation lossPeers lsePeers inputs
  have frame := prep.2.2.2.2.trans saved
  have labelsEq : returned.context.labels=f.labels := congrArg (fun r : ForwardReturn => r.context.labels) f.returnIdentity
  have coupled := int_read_region encode f.labelsReady before f.labels.region row
    (fun j => (between.memory f.labels.region j (Or.inr (Or.inr (Or.inl (congrArg Tensor.region labelsEq.symm))))).trans
      (frame.memory _ j k.labelsSnapshot.materialized.live)) read
  simpa only [labelsEq] using coupled

theorem backward_label_coupling {α σ : Type} [Inhabited α] (encode : α → Word)
    {obs : Observations α σ} {rt : Runtime α σ} {objects : ObjectRegistry rt} {M : FP32 α} {c : Context}
    {callerUpstream output : Tensor} {before after : World α σ}
    {b : BackwardData M rt c callerUpstream output before after} (k : BackwardKernelData obs objects b)
    (saved : before.live c.labels.region) (row : Nat) (read : IntRead encode before c.labels.region row) :
    IntRead encode k.launch.ready c.labels.region row :=
  int_read_region encode before k.launch.ready c.labels.region row (fun j => k.initialFrame.memory _ j saved) read

#axiomsClean snapshot_int_reads
#axiomsClean forward_label_coupling
#axiomsClean saved_label_coupling
#axiomsClean backward_label_coupling
#specNonCircular CloneBytes avoiding [Source.forward,Source.backward,Grid.forward,Grid.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Physical
