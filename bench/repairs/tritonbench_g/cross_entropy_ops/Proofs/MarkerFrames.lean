import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.MarkerPrimitives

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Resident
open VeriTile.Triton
open CrossEntropy1.Host (World Observations TensorLibrary IntegerViews CollectiveLibrary ObjectRegistry)
open CrossEntropy1.Physical (Layout Byte Retains OutsideView outside_cell)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- All bytes of retained live caller storage, including view holes and a
partial trailing FP32 word, follow from the actual marker source run. -/
theorem forward_marker_byte {α σ : Type} (layout : Layout σ) (observe : σ → Nat → Nat → Byte)
    (kept : RegionName → Prop) (integer : World Marker σ → Tensor → Nat → Int)
    (rt : Host.Runtime Marker σ) (gs gr : Nat → Nat)
    (rules : MarkerPrimitiveRules integer rt gs gr)
    {o : Host.Options} {logits labels : Tensor} {initial : World α σ} {final : World Marker σ}
    {returned : Host.ForwardReturn}
    {e : Host.ForwardData markerFP rt o logits labels (markWorld initial) returned final}
    (k : Host.ForwardKernelData (markerObservations integer) e)
    (labelDtype : labels.dtype="torch.int64") (size : e.world>1 → e.world=gs (o.processGroup.getD 0))
    (aggregation : Host.ForwardAggregation e) (lossPeers lsePeers : Nat → Marker)
    (inputs : Host.ForwardInputs rules.collectives aggregation lossPeers lsePeers)
    (rep : MarkerRep layout observe initial.native final)
    (lifetime : Retains layout kept initial.native final.native)
    (r : RegionName) (held : kept r) (live : initial.live r) (active : layout.active initial.native r)
    (offset : Nat) (inside : offset<(layout.buffer initial.native r).capacity) :
    observe final.native (layout.buffer initial.native r).device ((layout.buffer initial.native r).base+offset)=
      observe initial.native (layout.buffer initial.native r).device ((layout.buffer initial.native r).base+offset) := by
  have frame := Host.forward_return_frame (markerObservations integer) markerHost rt rules.library rules.arithmetic
    rules.zeros rules.integers gs gr rules.collectives k labelDtype size aggregation lossPeers lsePeers inputs
  exact rep.retained_byte r offset (frame.live r live) (lifetime.buffer r held active) inside (frame.memory r _ live)

theorem backward_marker_byte {α σ : Type} (layout : Layout σ) (observe : σ → Nat → Nat → Byte)
    (kept : RegionName → Prop) (integer : World Marker σ → Tensor → Nat → Int)
    (rt : Host.Runtime Marker σ) (lib : Host.LocalAPI (markerObservations integer) rt)
    {objects : ObjectRegistry rt.toRuntime} {c : Host.Context} {upstream output : Tensor}
    {initial : World α σ} {final : World Marker σ}
    {e : Host.BackwardData markerFP rt c upstream output (markWorld initial) final}
    (k : Host.BackwardKernelData (markerObservations integer) objects e)
    (rep : MarkerRep layout observe initial.native final)
    (lifetime : Retains layout kept initial.native final.native)
    (r : RegionName) (held : kept r) (live : initial.live r) (active : layout.active initial.native r)
    (offset : Nat) (inside : offset<(layout.buffer initial.native r).capacity)
    (outsideWork : OutsideView e.work r offset) (outsideOutput : OutsideView output r offset) :
    observe final.native (layout.buffer initial.native r).device ((layout.buffer initial.native r).base+offset)=
      observe initial.native (layout.buffer initial.native r).device ((layout.buffer initial.native r).base+offset) := by
  exact rep.retained_byte r offset
    (Host.backward_return_live (markerObservations integer) rt lib k r live)
    (lifetime.buffer r held active) inside
    (Host.backward_return_frame (markerObservations integer) rt lib k r (offset/4) live
      (outside_cell _ _ _ outsideWork) (outside_cell _ _ _ outsideOutput))

theorem backward_out_of_place_marker_byte {α σ : Type} (layout : Layout σ) (observe : σ → Nat → Nat → Byte)
    (kept : RegionName → Prop) (integer : World Marker σ → Tensor → Nat → Int)
    (rt : Host.Runtime Marker σ) (lib : Host.LocalAPI (markerObservations integer) rt)
    {objects : ObjectRegistry rt.toRuntime} {c : Host.Context} {upstream output : Tensor}
    {initial : World α σ} {final : World Marker σ}
    {e : Host.BackwardData markerFP rt c upstream output (markWorld initial) final}
    (k : Host.BackwardKernelData (markerObservations integer) objects e)
    (rep : MarkerRep layout observe initial.native final)
    (lifetime : Retains layout kept initial.native final.native)
    (logitsLive : initial.live c.logits.region) (outOfPlace : c.options.inplace≠Bool.true)
    (r : RegionName) (held : kept r) (live : initial.live r) (active : layout.active initial.native r)
    (offset : Nat) (inside : offset<(layout.buffer initial.native r).capacity) :
    observe final.native (layout.buffer initial.native r).device ((layout.buffer initial.native r).base+offset)=
      observe initial.native (layout.buffer initial.native r).device ((layout.buffer initial.native r).base+offset) := by
  exact rep.retained_byte r offset
    (Host.backward_return_live (markerObservations integer) rt lib k r live)
    (lifetime.buffer r held active) inside
    (Host.backward_out_of_place_frame (markerObservations integer) rt lib k logitsLive outOfPlace r _ live)

#axiomsClean forward_marker_byte
#axiomsClean backward_marker_byte
#axiomsClean backward_out_of_place_marker_byte
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Resident
