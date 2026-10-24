import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Lifetimes
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.BackwardFrames

namespace VeriTile.Bench.Repairs.CrossEntropy1.Physical
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000

theorem byte_same_cell {α σ : Type} [Inhabited α] (layout : Layout σ) (encode : α → Word)
    (a b : World α σ) (r : RegionName) (offset : Nat) (activeA : layout.active a.native r)
    (activeB : layout.active b.native r) (sameBuffer : layout.buffer a.native r=layout.buffer b.native r)
    (inside : offset<(layout.buffer a.native r).capacity)
    (same : a.memory r (offset/4)=b.memory r (offset/4)) :
    bytes layout encode a (layout.buffer a.native r).device ((layout.buffer a.native r).base+offset)=
      bytes layout encode b (layout.buffer a.native r).device ((layout.buffer a.native r).base+offset) := by
  let j : Fin 4 := ⟨offset%4,by omega⟩
  have coordinate : 4*(offset/4)+j.val=offset := by dsimp [j]; omega
  have ha := raw_bytes layout encode a r (offset/4) j activeA (by simpa only [coordinate] using inside)
  have hb := raw_bytes layout encode b r (offset/4) j activeB (by simpa only [← sameBuffer,coordinate] using inside)
  rw [← sameBuffer] at hb
  have equalWord : rawWord encode a r (offset/4)=rawWord encode b r (offset/4) := by simp only [rawWord,same]
  rw [equalWord] at ha
  simpa only [Nat.add_assoc,coordinate] using ha.trans hb.symm

/-- Byte offset outside the actual FP32 coordinates of a view in one caller
allocation. Storage aliases share the region; row holes and offsets are kept. -/
def OutsideView (t : Tensor) (r : RegionName) (offset : Nat) : Prop :=
  r≠t.region ∨ ∀ j, j<t.numel → offset<4*t.index j ∨ 4*t.index j+4≤offset

theorem outside_cell (t : Tensor) (r : RegionName) (offset : Nat) (outside : OutsideView t r offset) :
    r≠t.region ∨ ∀ j, j<t.numel → offset/4≠t.index j := by
  rcases outside with different|outside
  · exact Or.inl different
  · right
    intro j hj same
    have gap := outside j hj
    omega

/-- Every byte of each externally retained caller allocation survives the
actual forward wrapper, including aggregation and save-for-backward. -/
theorem forward_caller_bytes {α σ : Type} [Inhabited α] (layout : Layout σ) (encode : α → Word)
    (kept : RegionName → Prop) (obs : Observations α σ) (A : TensorArithmetic α) (rt : Runtime α σ)
    (life : LifetimeLibrary layout rt kept) (lib : Library obs rt) (ops : TensorLibrary obs A rt)
    (ints : IntegerViews obs) (gs gr : Nat → Nat) (comms : CollectiveLibrary obs A rt gs gr)
    {M : FP32 α} {o : Options} {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    {f : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs f)
    (labelDtype : labels.dtype="torch.int64") (size : f.world>1 → f.world=gs (o.processGroup.getD 0))
    (aggregation : ForwardAggregation f) (lossPeers lsePeers : Nat → α)
    (inputs : ForwardInputs comms aggregation lossPeers lsePeers)
    (r : RegionName) (retained : kept r) (live : initial.live r) (active : layout.active initial.native r)
    (offset : Nat) (inside : offset<(layout.buffer initial.native r).capacity) :
    bytes layout encode final (layout.buffer initial.native r).device ((layout.buffer initial.native r).base+offset)=
      bytes layout encode initial (layout.buffer initial.native r).device ((layout.buffer initial.native r).base+offset) := by
  have logical := (forward_values obs A rt lib ops ints gs gr comms k labelDtype size aggregation lossPeers lsePeers inputs).2.2.2.2
  have lifetime := forward_lifetime layout rt kept life f
  exact preserved_allocation_bytes layout encode obs initial final r logical live active
    (lifetime.active r retained active) (lifetime.buffer r retained active).symm offset inside

/-- Complete caller-byte frame after actual backward and optional copyback.
The only exclusions are bytes of the actual work and returned destination
views. This also covers caller views created from saved storage between calls. -/
theorem backward_caller_bytes {α σ : Type} [Inhabited α] (layout : Layout σ) (encode : α → Word)
    (kept : RegionName → Prop) (obs : Observations α σ) (rt : Runtime α σ)
    (life : LifetimeLibrary layout rt kept) (lib : Library obs rt) {objects : ObjectRegistry rt}
    {M : FP32 α} {c : Context} {callerUpstream output : Tensor} {initial final : World α σ}
    {b : BackwardData M rt c callerUpstream output initial final} (k : BackwardKernelData obs objects b)
    (r : RegionName) (retained : kept r) (live : initial.live r) (active : layout.active initial.native r)
    (offset : Nat) (inside : offset<(layout.buffer initial.native r).capacity)
    (outsideWork : OutsideView b.work r offset) (outsideOutput : OutsideView output r offset) :
    bytes layout encode final (layout.buffer initial.native r).device ((layout.buffer initial.native r).base+offset)=
      bytes layout encode initial (layout.buffer initial.native r).device ((layout.buffer initial.native r).base+offset) := by
  have logical := backward_return_frame obs rt lib k r (offset/4) live
    (outside_cell b.work r offset outsideWork) (outside_cell output r offset outsideOutput)
  have lifetime := backward_lifetime layout rt kept life b
  exact (byte_same_cell layout encode initial final r offset active (lifetime.active r retained active)
    (lifetime.buffer r retained active).symm inside logical.symm).symm

/-- In out-of-place backward the two destination allocations are proved
fresh, so no byte of any pre-existing caller allocation is excluded. -/
theorem backward_out_of_place_bytes {α σ : Type} [Inhabited α] (layout : Layout σ) (encode : α → Word)
    (kept : RegionName → Prop) (obs : Observations α σ) (rt : Runtime α σ)
    (life : LifetimeLibrary layout rt kept) (lib : Library obs rt) {objects : ObjectRegistry rt}
    {M : FP32 α} {c : Context} {callerUpstream output : Tensor} {initial final : World α σ}
    {b : BackwardData M rt c callerUpstream output initial final} (k : BackwardKernelData obs objects b)
    (logitsLive : initial.live c.logits.region) (outOfPlace : c.options.inplace≠Bool.true)
    (r : RegionName) (retained : kept r) (live : initial.live r) (active : layout.active initial.native r)
    (offset : Nat) (inside : offset<(layout.buffer initial.native r).capacity) :
    bytes layout encode final (layout.buffer initial.native r).device ((layout.buffer initial.native r).base+offset)=
      bytes layout encode initial (layout.buffer initial.native r).device ((layout.buffer initial.native r).base+offset) := by
  have fresh := backward_out_of_place_fresh obs rt lib k logitsLive outOfPlace
  exact backward_caller_bytes layout encode kept obs rt life lib k r retained live active offset inside
    (Or.inl (fun same => fresh.2 (same ▸ live))) (Or.inl (fun same => fresh.1 (same ▸ live)))

#axiomsClean byte_same_cell
#axiomsClean forward_caller_bytes
#axiomsClean backward_caller_bytes
#axiomsClean backward_out_of_place_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Physical
