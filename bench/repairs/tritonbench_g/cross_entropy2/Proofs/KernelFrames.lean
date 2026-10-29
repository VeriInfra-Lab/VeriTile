import bench.repairs.tritonbench_g.cross_entropy2.Proofs.LocalFrames

namespace VeriTile.Bench.Repairs.CrossEntropy2.Frames
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Step bind Observations IntegerViews)
open VeriTile.Bench.Repairs.CrossEntropy1.Physical (Layout Retains)
open VeriTile.Bench.Repairs.CrossEntropy1.Grid (Address flat unflat)
noncomputable section
set_option maxHeartbeats 6000000

theorem forward_other {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters) (m : Memory α)
    (count : Nat) (r : RegionName) (i : Nat) (loss : r≠p.loss) (lse : r≠p.lse) (z : r≠p.zLoss) :
    Grid.forward M p m count r i=m r i := by
  have preserved : ∀ schedule current,
      VeriTile.Bench.Repairs.CrossEntropy1.Schedule.run (Grid.step M p) schedule current (r,i)=current (r,i) := by
    intro schedule
    induction schedule with
    | nil => intro current; rfl
    | cons q tail ih =>
      intro current
      change VeriTile.Bench.Repairs.CrossEntropy1.Schedule.run (Grid.step M p) tail (Grid.step M p q current) (r,i)=_
      rw [ih]
      exact Grid.frame M p q current (r,i) (by
        rintro (⟨h,_⟩|⟨h,_⟩|⟨h,_⟩)
        · exact loss h
        · exact lse h
        · exact z h)
  exact preserved _ (flat m)

theorem backward_other {α : Type} [Inhabited α] (M : FP32 α) (p : Parameters) (m : Memory α)
    (count : Nat) (r : RegionName) (i : Nat) (gradient : r≠p.gradient) :
    backwardGrid M p m count r i=m r i := by
  rw [backward_models]
  have preserved : ∀ schedule current,
      VeriTile.Bench.Repairs.CrossEntropy1.Schedule.run (BackwardGrid.backwardStep M p) schedule current (r,i)=current (r,i) := by
    intro schedule
    induction schedule with
    | nil => intro current; rfl
    | cons q tail ih =>
      intro current
      change VeriTile.Bench.Repairs.CrossEntropy1.Schedule.run (BackwardGrid.backwardStep M p) tail (BackwardGrid.backwardStep M p q current) (r,i)=_
      rw [ih]
      exact BackwardGrid.backward_frame M p q current (r,i) (by
        rintro ⟨lane,active,equal⟩
        exact gradient (congrArg Prod.fst equal))
  exact preserved _ (flat m)

variable {α σ : Type} [Inhabited α] {obs : Observations α σ} {layout : Layout σ} {kept : RegionName → Prop}
  {rt : Runtime α σ}

theorem grid_frame (ints : IntegerViews obs) (old ready : World α σ)
    (frame : Frame obs layout kept old ready) (kernel : Memory α → Memory α)
    (unchanged : ∀ r, old.live r → ∀ i, kernel ready.memory r i=ready.memory r i) :
    Frame obs layout kept old {ready with memory:=kernel ready.memory} := by
  refine ⟨⟨frame.storage.live,?_,?_⟩,frame.lifetime⟩
  · intro r i live
    exact (unchanged r live i).trans (frame.storage.memory r i live)
  · intro t i live
    exact (ints.localRead {ready with memory:=kernel ready.memory} ready t i rfl (unchanged t.region live)).trans
      (frame.storage.integers t i live)

theorem forward_launch_frame (lib : LocalLibrary obs layout kept rt) (ints : IntegerViews obs)
    (M : FP32 α) (p : Parameters) (count : Nat) (old before : World α σ)
    (frame : Frame obs layout kept old before)
    (lossFresh : ¬old.live p.loss) (lseFresh : ¬old.live p.lse) (zFresh : ¬old.live p.zLoss) :
    Post (fun _ after => Frame obs layout kept old after) (Frame obs layout kept old) (forwardLaunch M rt p count before) := by
  unfold forwardLaunch
  split
  · exact frame
  · apply bind_post _ _ _ _ _ (setup_frame lib old before frame _ (.forward p count))
    intro _ ready readyFrame
    apply grid_frame ints old ready readyFrame (fun m => Grid.forward M p m count)
    intro r live i
    exact forward_other M p ready.memory count r i (fun eq => lossFresh (eq ▸ live))
      (fun eq => lseFresh (eq ▸ live)) (fun eq => zFresh (eq ▸ live))

theorem backward_launch_frame (lib : LocalLibrary obs layout kept rt) (ints : IntegerViews obs)
    (M : FP32 α) (p : Parameters) (count : Nat) (old before : World α σ)
    (frame : Frame obs layout kept old before) (fresh : ¬old.live p.gradient) :
    Post (fun _ after => Frame obs layout kept old after) (Frame obs layout kept old) (backwardLaunch M rt p count before) := by
  unfold backwardLaunch
  split
  · exact frame
  · apply bind_post _ _ _ _ _ (setup_frame lib old before frame _ (.backward p count))
    intro _ ready readyFrame
    apply grid_frame ints old ready readyFrame (fun m => backwardGrid M p m count)
    intro r live i
    exact backward_other M p ready.memory count r i (fun eq => fresh (eq ▸ live))

#axiomsClean forward_other
#axiomsClean backward_other
#axiomsClean forward_launch_frame
#axiomsClean backward_launch_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Frames
