import bench.repairs.tritonbench_g.cross_entropy1.Proofs.RoleBindings

namespace VeriTile.Bench.Repairs.CrossEntropy1.Physical
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 10000000

structure ReadAt {σ : Type} (layout : Layout σ) (rights : Permissions σ) (native : σ)
    (t : Tensor) (width : Nat) (pointer : BitVec 64) (index : Nat) : Prop where
  address : (Native.address (layout.buffer native t.region).base width pointer).toNat=
    (layout.buffer native t.region).base+width*index
  readable : ∀ byte, byte<width → rights.readable native t.device ((layout.buffer native t.region).base+width*index+byte)

structure WriteAt {σ : Type} (layout : Layout σ) (rights : Permissions σ) (native : σ)
    (t : Tensor) (width : Nat) (pointer : BitVec 64) (index : Nat) : Prop where
  address : (Native.address (layout.buffer native t.region).base width pointer).toNat=
    (layout.buffer native t.region).base+width*index
  writable : ∀ byte, byte<width → rights.writable native t.device ((layout.buffer native t.region).base+width*index+byte)

/-- Derived saved metadata and active typed reads at this backward launch.
Every field is proved from the preceding successful forward and current
successful autograd entry in `backward_prepared`. -/
structure PreparedBackward {α σ : Type} [Inhabited α] (encode : α → Word)
    (obs : Observations α σ) {M : FP32 α} {rt : Runtime α σ} (objects : ObjectRegistry rt)
    {c : Context} {upstream output : Tensor} {before after : World α σ}
    (b : BackwardData M rt c upstream output before after) where
  kernel : BackwardKernelData obs objects b
  logitsLayout : RowLayout c.logits
  labelsShape : c.labels.shape=[b.rows]
  lseShape : c.lse.shape=[b.rows]
  labelsDense : c.labels.Dense
  lseDense : c.lse.Dense
  labelsRead : ∀ row, row<b.rows → IntRead encode kernel.launch.ready c.labels.region row

theorem backward_prepared {α σ : Type} [Inhabited α] (encode : α → Word)
    (obs : Observations α σ) (M : FP32 α) (A : TensorArithmetic α) (call : Entry.Invocation obs M)
    (ops : TensorLibrary obs A call.runtime) (ints : IntegerViews obs) (clone : CloneBytes encode obs call.runtime)
    (comms : CollectiveLibrary obs A call.runtime call.groups.size call.groups.rank)
    (labelDtype : call.labels.dtype="torch.int64") (aggregation : ForwardAggregation call.data)
    (lossPeers lsePeers : Nat → α) (inputs : ForwardInputs comms aggregation lossPeers lsePeers)
    (versions : σ → Nat → Nat) (before after : World α σ)
    (between : SavedPreserves obs call.returned.context call.saved before) (upstream output : Tensor)
    (upstreamLive : before.live upstream.region)
    (success : Autograd.backwardEntry M call.runtime versions
      (Autograd.capture versions call.returned.context call.saved.native) upstream before=.ok output after) :
    Nonempty (PreparedBackward encode obs call.objects (call.backwardData versions upstream output before after success)) := by
  let b := call.backwardData versions upstream output before after success
  have values := forward_values obs A call.runtime call.library ops ints call.groups.size call.groups.rank comms call.kernel
    labelDtype call.size aggregation lossPeers lsePeers inputs
  have saved := forward_backward_context obs A call.runtime call.library ops ints call.objects call.groups.size call.groups.rank comms
    call.kernel call.views labelDtype call.size aggregation lossPeers lsePeers inputs
  have ctx := between.context saved
  obtain ⟨kernel⟩ := backward_kernel_data obs call.runtime call.library call.objects b ctx upstreamLive
  have logitsEq : call.returned.context.logits=call.data.workingLogits :=
    congrArg (fun r : ForwardReturn => r.context.logits) call.data.returnIdentity
  have labelsEq : call.returned.context.labels=call.data.labels :=
    congrArg (fun r : ForwardReturn => r.context.labels) call.data.returnIdentity
  have lseEq : call.returned.context.lse=call.data.pair.2 :=
    congrArg (fun r : ForwardReturn => r.context.lse) call.data.returnIdentity
  have shapes : [b.rows,b.cols]=[call.data.rows,call.data.cols] :=
    b.shape.symm.trans (values.2.2.2.1.metadata.shape.trans call.data.logitsShape)
  have dimensions : b.rows=call.data.rows ∧ b.cols=call.data.cols := by simpa only [List.cons.injEq,and_true] using shapes
  have dense := forward_saved_lse obs A call.runtime call.library ops ints call.groups.size call.groups.rank comms
    call.kernel call.size aggregation
  refine ⟨⟨kernel,by simpa only [logitsEq] using call.kernel.layout,
    by simpa only [←dimensions.1] using values.2.2.1.shapeEq,
    by simpa only [←dimensions.1] using values.2.1.shapeEq,
    by simpa only [labelsEq] using call.kernel.labelsSnapshot.dense,
    by simpa only [lseEq] using dense.1,?_⟩⟩
  intro row hr
  have read := saved_label_coupling encode obs A call.runtime call.library clone ops ints call.groups call.groups.size call.groups.rank
    comms call.kernel call.logitsLive call.labelsLive call.rank labelDtype call.size aggregation lossPeers lsePeers inputs between
    row (by change row<b.rows at hr; simpa only [dimensions.1] using hr)
  exact backward_label_coupling encode kernel ctx.labelsLive row read

/-- Every backward pointer role, for an actual launch obtained above. Saved
label raw-byte coupling is retained in `ready.labelsRead`; output and logit
pointers keep their actual strides, storage offsets, and tensor identities. -/
theorem backward_roles {α σ : Type} [Inhabited α] (layout : Layout σ) (rights : Permissions σ)
    (encode : α → Word) (obs : Observations α σ) {M : FP32 α} {rt : Runtime α σ} {objects : ObjectRegistry rt}
    {c : Context} {upstream output : Tensor} {before after : World α σ}
    {b : BackwardData M rt c upstream output before after} (ready : PreparedBackward encode obs objects b)
    (args : BackwardArguments layout rights ready.kernel) (upstreamShape : upstream.shape=[b.rows])
    (row block lane : Nat) (hr : row<b.rows) (hb : block<b.count) (hl : lane<b.parameters.block)
    (active : column b.parameters block lane<b.cols) :
    ReadAt layout rights ready.kernel.launch.ready.native c.logits 4
      (Native.element b.parameters.logitsBase row b.parameters.logitsStride b.parameters.block block lane)
      (logitAddress b.parameters row (column b.parameters block lane)) ∧
    ReadAt layout rights ready.kernel.launch.ready.native c.labels 8
      (BitVec.ofNat 64 (b.parameters.labelsBase+row)) (b.parameters.labelsBase+row) ∧
    ReadAt layout rights ready.kernel.launch.ready.native c.lse 4
      (BitVec.ofNat 64 (b.parameters.lseBase+row)) (b.parameters.lseBase+row) ∧
    ReadAt layout rights ready.kernel.launch.ready.native b.upstream 4
      (BitVec.ofNat 64 (b.parameters.upstreamBase+row*b.parameters.upstreamStride))
      (b.parameters.upstreamBase+row*b.parameters.upstreamStride) ∧
    WriteAt layout rights ready.kernel.launch.ready.native b.work 4
      (Native.element b.parameters.gradientBase row b.parameters.gradientStride b.parameters.block block lane)
      (gradientAddress b.parameters row (column b.parameters block lane)) := by
  have logits := backward_logit_pointer layout rights ready.kernel args ready.logitsLayout row block lane hr hb hl active
  have labels := dense_read_pointer layout rights ready.kernel.launch.ready.native c.labels 8 b.rows row args.labels
    (by decide) ready.labelsShape ready.labelsDense hr
  have lse := dense_read_pointer layout rights ready.kernel.launch.ready.native c.lse 4 b.rows row args.lse
    (by decide) ready.lseShape ready.lseDense hr
  have upstream := backward_upstream_pointer layout rights ready.kernel args upstreamShape row hr
  have output := backward_store_pointer layout rights ready.kernel args row block lane hr hb hl active
  exact ⟨⟨logits.1,logits.2⟩,⟨labels.1,labels.2⟩,⟨lse.1,lse.2⟩,⟨upstream.1,upstream.2⟩,⟨output.1,output.2⟩⟩

/-- Nonignored label subtraction uses the actual rank-query result and
global class count of this forward call. Ignored labels take the separately
proved safe replacement path, including INT64_MIN. -/
theorem forward_label_shift {α σ : Type} [Inhabited α] (obs : Observations α σ) (M : FP32 α)
    (call : Entry.Invocation obs M) (abi : ScalarABI call.data.parameters call.data.count)
    (labelDtype : call.labels.dtype="torch.int64") (row : Nat) (hr : row<call.data.rows)
    (valid : 0≤obs.integer call.initial call.labels row ∧
      obs.integer call.initial call.labels row<(call.data.world*call.data.cols : Nat)) :
    (Native.shiftedLabel (Source.label call.data.parameters call.kernel.launch.ready.memory row)
      call.data.parameters.classStart).toInt=
      obs.integer call.initial call.labels row-call.data.parameters.classStart := by
  have group := forward_group_metadata obs call.runtime call.groups call.data
  have positive : 0<call.data.cols := by
    have nonzero := call.data.nonzero
    by_contra no
    have zero : call.data.cols=0 := by omega
    simp [zero,fwdWidth,nextPower] at nonzero
  have total : 0<call.data.world*call.data.cols := Nat.mul_pos group.1 positive
  have start : call.data.rank*call.data.cols<call.data.world*call.data.cols := Nat.mul_lt_mul_of_pos_right group.2.1 positive
  have label := forward_kernel_labels obs call.kernel labelDtype row hr
  rw [label]
  apply Native.shifted_label _ _ (call.data.world*call.data.cols : Nat)
  · exact ⟨by exact_mod_cast total,by exact_mod_cast abi.total⟩
  · exact valid
  · constructor
    · exact Int.natCast_nonneg _
    · change ((call.data.rank*call.data.cols : Nat) : Int)<(call.data.world*call.data.cols : Nat)
      exact_mod_cast start

#axiomsClean forward_label_shift
#axiomsClean backward_prepared
#axiomsClean backward_roles
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Physical
