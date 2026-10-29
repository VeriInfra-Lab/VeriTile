import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Entries
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.CallerBytes

namespace VeriTile.Bench.Repairs.CrossEntropy1.Physical
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

structure CallerFrames {α σ : Type} [Inhabited α] (layout : Layout σ) (encode : α → Word)
    (kept : RegionName → Prop) (initial saved before after : World α σ) (inplace : Bool) (work output : Tensor) : Prop where
  forward : ∀ r, kept r → initial.live r → layout.active initial.native r → ∀ offset,
    offset<(layout.buffer initial.native r).capacity →
    bytes layout encode saved (layout.buffer initial.native r).device ((layout.buffer initial.native r).base+offset)=
      bytes layout encode initial (layout.buffer initial.native r).device ((layout.buffer initial.native r).base+offset)
  backward : ∀ r, kept r → before.live r → layout.active before.native r → ∀ offset,
    offset<(layout.buffer before.native r).capacity → OutsideView work r offset → OutsideView output r offset →
    bytes layout encode after (layout.buffer before.native r).device ((layout.buffer before.native r).base+offset)=
      bytes layout encode before (layout.buffer before.native r).device ((layout.buffer before.native r).base+offset)
  outOfPlace : inplace≠Bool.true → ∀ r, kept r → before.live r → layout.active before.native r → ∀ offset,
    offset<(layout.buffer before.native r).capacity →
    bytes layout encode after (layout.buffer before.native r).device ((layout.buffer before.native r).base+offset)=
      bytes layout encode before (layout.buffer before.native r).device ((layout.buffer before.native r).base+offset)

/-- Complete backing-allocation frames from actual successful invocations.
All bytes are covered, including holes and partial trailing words; views
created from saved storage before backward are included in the later frame. -/
theorem entry_frames {α σ : Type} [Inhabited α] (layout : Layout σ) (encode : α → Word)
    (kept : RegionName → Prop) (obs : Observations α σ) (M : FP32 α) (A : TensorArithmetic α)
    (call : Entry.Invocation obs M) (ops : TensorLibrary obs A call.runtime) (ints : IntegerViews obs)
    (life : LifetimeLibrary layout call.runtime kept)
    (comms : CollectiveLibrary obs A call.runtime call.groups.size call.groups.rank)
    (labelDtype : call.labels.dtype="torch.int64") (aggregation : ForwardAggregation call.data)
    (lossPeers lsePeers : Nat → α) (inputs : ForwardInputs comms aggregation lossPeers lsePeers)
    (versions : σ → Nat → Nat) (before after : World α σ)
    (between : SavedPreserves obs call.returned.context call.saved before) (upstream output : Tensor)
    (upstreamLive : before.live upstream.region)
    (success : Autograd.backwardEntry M call.runtime versions
      (Autograd.capture versions call.returned.context call.saved.native) upstream before=.ok output after) :
    CallerFrames layout encode kept call.initial call.saved before after call.returned.context.options.inplace
      (call.backwardData versions upstream output before after success).work output := by
  have saved := forward_backward_context obs A call.runtime call.library ops ints call.objects call.groups.size call.groups.rank comms
    call.kernel call.views labelDtype call.size aggregation lossPeers lsePeers inputs
  have ctx := between.context saved
  obtain ⟨kernel⟩ := backward_kernel_data obs call.runtime call.library call.objects
    (call.backwardData versions upstream output before after success) ctx upstreamLive
  refine ⟨?_,?_,?_⟩
  · exact forward_caller_bytes layout encode kept obs A call.runtime life call.library ops ints
      call.groups.size call.groups.rank comms call.kernel labelDtype call.size aggregation lossPeers lsePeers inputs
  · exact backward_caller_bytes layout encode kept obs call.runtime life call.library kernel
  · intro outOfPlace
    exact backward_out_of_place_bytes layout encode kept obs call.runtime life call.library kernel ctx.logitsLive outOfPlace

#axiomsClean entry_frames
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Physical
